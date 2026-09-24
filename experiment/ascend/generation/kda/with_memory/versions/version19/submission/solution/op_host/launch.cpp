#include "op_host/launch.h"
#include "aclnn_chunk_kda_fwd.h"

#include <limits>

namespace {

constexpr int64_t kDim = 128;
constexpr int64_t kBatch = 1;
constexpr int64_t kTokens = 4096;
constexpr int64_t kHeads = 96;

class Descriptor {
public:
    explicit Descriptor(const torch::Tensor& tensor) {
        const auto shape = tensor.sizes().vec();
        const auto strides = tensor.strides().vec();
        const auto dtype = tensor.scalar_type() == at::kBFloat16 ? ACL_BF16 : ACL_FLOAT;
        value_ = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), tensor.data_ptr());
        TORCH_CHECK(value_ != nullptr, "aclCreateTensor failed");
    }

    ~Descriptor() {
        if (value_ != nullptr) aclDestroyTensor(value_);
    }

    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;

    aclTensor* get() const { return value_; }

private:
    aclTensor* value_ = nullptr;
};

void require(const torch::Tensor& tensor, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == dtype, name, ": wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, ": must be contiguous");
}

void require_shape(const torch::Tensor& tensor, std::initializer_list<int64_t> shape,
                   const char* name) {
    TORCH_CHECK(tensor.sizes() == torch::IntArrayRef(shape), name, ": wrong shape");
}

}  // namespace

void launch(const std::vector<torch::Tensor>& inputs,
            const std::vector<torch::Tensor>& outputs, aclrtStream stream) {
    TORCH_CHECK(inputs.size() == 10, "kda expects ten inputs");
    TORCH_CHECK(outputs.size() == 2, "kda expects two outputs");
    const torch::Tensor& q = inputs[0];
    const torch::Tensor& k = inputs[1];
    const torch::Tensor& v = inputs[2];
    const torch::Tensor& g = inputs[3];
    const torch::Tensor& beta = inputs[4];
    const torch::Tensor& scale = inputs[5];
    const torch::Tensor& a_log = inputs[6];
    const torch::Tensor& dt_bias = inputs[7];
    const torch::Tensor& lower_bound = inputs[8];
    const torch::Tensor& initial_state = inputs[9];
    torch::Tensor output = outputs[0];
    torch::Tensor final_state = outputs[1];

    TORCH_CHECK(q.dim() == 4, "q must be [batch, tokens, heads, dim]");
    const int64_t batch = q.size(0);
    const int64_t tokens = q.size(1);
    const int64_t heads = q.size(2);
    TORCH_CHECK(batch == kBatch && tokens == kTokens && heads == kHeads &&
                q.size(3) == kDim, "unsupported KDA shape");

    for (const auto& tensor : inputs) {
        TORCH_CHECK(tensor.device() == q.device(), "all inputs must be on q's device");
    }
    for (const auto& tensor : outputs) {
        TORCH_CHECK(tensor.device() == q.device(), "all outputs must be on q's device");
    }

    for (const auto* pair : {&q, &k, &v, &g}) require(*pair, at::kBFloat16, "q/k/v/g");
    for (const auto* pair : {&q, &k, &v, &g}) {
        require_shape(*pair, {batch, tokens, heads, kDim}, "q/k/v/g");
    }
    require(output, at::kBFloat16, "output");
    require_shape(output, {batch, tokens, heads, kDim}, "output");
    require(beta, at::kFloat, "beta");
    require_shape(beta, {batch, tokens, heads}, "beta");
    require(scale, at::kFloat, "scale");
    require_shape(scale, {}, "scale");
    require(a_log, at::kFloat, "a_log");
    require_shape(a_log, {heads}, "a_log");
    require(dt_bias, at::kFloat, "dt_bias");
    require_shape(dt_bias, {heads, kDim}, "dt_bias");
    require(lower_bound, at::kFloat, "lower_bound");
    require_shape(lower_bound, {}, "lower_bound");
    require(initial_state, at::kFloat, "initial_state");
    require_shape(initial_state, {batch, heads, kDim, kDim}, "initial_state");
    require(final_state, at::kFloat, "final_state");
    require_shape(final_state, {batch, heads, kDim, kDim}, "final_state");

    Descriptor query_desc(q);
    Descriptor key_desc(k);
    Descriptor value_desc(v);
    Descriptor gate_desc(g);
    Descriptor beta_desc(beta);
    Descriptor scale_desc(scale);
    Descriptor a_log_desc(a_log);
    Descriptor dt_bias_desc(dt_bias);
    Descriptor lower_desc(lower_bound);
    // The device kernel keeps the recurrent state as [key, value] row tiles, so
    // it is given a transposed copy and the produced state is transposed back.
    auto state_kv = initial_state.transpose(2, 3).contiguous();
    auto final_kv = torch::empty_like(state_kv);

    Descriptor state_desc(state_kv);
    Descriptor output_desc(output);
    Descriptor final_desc(final_kv);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto query_status = aclnnChunkKdaFwdGetWorkspaceSize(
        query_desc.get(), key_desc.get(), value_desc.get(), gate_desc.get(),
        beta_desc.get(), scale_desc.get(), a_log_desc.get(), dt_bias_desc.get(),
        lower_desc.get(), state_desc.get(), output_desc.get(), final_desc.get(),
        &workspace_size, &executor);
    TORCH_CHECK(query_status == ACL_SUCCESS, "ACLNN workspace query failed: ", query_status);
    TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace too large");

    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                  q.options().dtype(at::kByte));
    const auto launch_status = aclnnChunkKdaFwd(
        workspace_size == 0 ? nullptr : workspace.data_ptr(), workspace_size,
        executor, stream);
    TORCH_CHECK(launch_status == ACL_SUCCESS, "ACLNN launch failed: ", launch_status);

    final_state.copy_(final_kv.transpose(2, 3));

    // Workspace allocation and ACLNN execution use the caller's stream, so
    // the caching allocator can reuse its storage in stream order.
}
