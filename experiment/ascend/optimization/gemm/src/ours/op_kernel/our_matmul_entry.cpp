#include "kernel_operator.h"
#include "our_matmul_kernel.h"

extern "C" __global__ __aicore__ void our_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c, OurTiling tiling)
{
    // Pure-cube kernel: only Cube cores start, no AIV participation.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    OurMatmul::Kernel kernel;
    AscendC::TPipe pipe;
    pipe.Destroy();  // no implicit cross-pipe sync inserted by the pipe
    kernel.Init(a, b, c, tiling);
    kernel.Process();
}
