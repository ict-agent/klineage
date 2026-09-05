#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace {

constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kAlignment = 8;

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    kAlignment,
    Accumulator,
    Accumulator>;
using Expert = cutlass::gemm::device::Gemm<
    Element,
    cutlass::layout::RowMajor,
    Element,
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    kAlignment,
    kAlignment>;

}  // namespace

extern "C" cudaError_t klineage_launch(
    const void* const* inputs,
    void* const* outputs,
    cudaStream_t stream) {
  if (!inputs || !outputs || !inputs[0] || !inputs[1] || !outputs[0]) {
    return cudaErrorInvalidValue;
  }

  auto const* x = static_cast<Element const*>(inputs[0]);
  auto const* weight = static_cast<Element const*>(inputs[1]);
  auto* output = static_cast<Element*>(outputs[0]);
  typename Expert::Arguments arguments(
      {kM, kN, kK},
      {x, kK},
      {weight, kK},
      {output, kN},
      {output, kN},
      {1.0F, 0.0F});

  if (Expert::can_implement(arguments) != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }

  Expert expert;
  if (expert(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorLaunchFailure;
  }
  return cudaSuccess;
}
