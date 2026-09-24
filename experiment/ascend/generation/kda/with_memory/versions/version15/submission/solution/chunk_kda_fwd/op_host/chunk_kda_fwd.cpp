#include "register/op_def_registry.h"
#include "chunk_kda_fwd_tiling.h"

namespace optiling {
ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto* input = context->GetInputShape(0);
    if (input == nullptr) return ge::GRAPH_FAILED;

    const auto& shape = input->GetStorageShape();
    constexpr int64_t kExpectedRank = 4;
    constexpr int64_t kExpectedBatch = 1;
    constexpr int64_t kExpectedTokens = 4096;
    constexpr int64_t kExpectedHeads = 96;
    constexpr int64_t kExpectedDim = 128;
    constexpr int64_t kInitialChunkSize = 64;
    constexpr int64_t kVBlocks = 2;
    if (shape.GetDimNum() != kExpectedRank || shape.GetDim(0) != kExpectedBatch ||
        shape.GetDim(1) != kExpectedTokens || shape.GetDim(2) != kExpectedHeads ||
        shape.GetDim(3) != kExpectedDim) return ge::GRAPH_FAILED;

    ChunkKdaFwdTilingData data;
    data.set_chunkSize(kInitialChunkSize);
    auto* buffer = context->GetRawTilingData();
    if (buffer == nullptr) return ge::GRAPH_FAILED;
    data.SaveToBuffer(buffer->GetData(), buffer->GetCapacity());
    buffer->SetDataSize(data.GetDataSize());
    context->SetBlockDim(kExpectedHeads * kVBlocks);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ChunkKdaFwd : public OpDef {
public:
    explicit ChunkKdaFwd(const char *name) : OpDef(name) {
        for (const char *input : {"q", "k", "v", "g"}) {
            this->Input(input).ParamType(REQUIRED)
                .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        }
        for (const char *input : {"beta", "scale", "a_log", "dt_bias",
                                 "lower_bound", "initial_state"}) {
            this->Input(input).ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        }
        this->Output("output").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("final_state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(ChunkKdaFwd);
}
