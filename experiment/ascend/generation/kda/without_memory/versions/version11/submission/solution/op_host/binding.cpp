#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "op_host/launch.h"

void kernel(torch::Tensor q, torch::Tensor k, torch::Tensor v,
            torch::Tensor g, torch::Tensor beta, torch::Tensor scale,
            torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor lower_bound,
            torch::Tensor initial_state, torch::Tensor output, torch::Tensor final_state) {
    std::vector<torch::Tensor> inputs{q, k, v, g, beta, scale, a_log, dt_bias,
                                      lower_bound, initial_state};
    std::vector<torch::Tensor> outputs{output, final_state};
    c10::DeviceGuard guard(q.device());
    auto stream = c10_npu::getCurrentNPUStream(q.get_device()).stream(true);
    launch(inputs, outputs, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("kernel", &kernel);
}
