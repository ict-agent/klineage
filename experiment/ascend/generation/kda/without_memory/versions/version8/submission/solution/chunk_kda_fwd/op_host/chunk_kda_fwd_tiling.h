#pragma once

#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkKdaFwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwd, ChunkKdaFwdTilingData)
}
