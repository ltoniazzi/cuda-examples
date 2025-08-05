#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}




// Softmax kernel
__global__ void softmax_tiled_kernel(float* V, float* O, int d) {
    const int TILE_SIZE = 32; 
    __shared__ float Ts[TILE_SIZE];
    
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIdxTile = threadIdx.x;

    float res = 0.0f;
    float tot = 0.0f;
    float max = -INFINITY;
    float max_prev = -INFINITY;
    float cur = 0.0f;


    int nTiles = cdiv(d, TILE_SIZE);

    if (rowIdx < d) {
        for (int nTile=0; nTile < nTiles; nTile++) {

            if (nTile*TILE_SIZE + rowIdxTile < d) {
                cur = V[nTile*TILE_SIZE + rowIdxTile];
                Ts[rowIdxTile] = cur;
                if (nTile*TILE_SIZE + rowIdxTile == rowIdx) {
                    res = cur;
                }
            }
            else {
                Ts[rowIdxTile] = 0.0f;
            }

            __syncthreads();

            for (int i=0; i < TILE_SIZE; i++) {
                max = fmaxf(max_prev, Ts[i]);
                if (max == max_prev) {
                    tot += __expf(Ts[i] - max);
                }
                else {
                    tot = tot * __expf(max_prev - max) + 1; // 1=__expf(Ts[i] - max) if max != max_prev
                    max_prev = max;
                }
            }
            __syncthreads();

        };


        O[rowIdx] = __expf(res-max)/tot; // (1 div + 1 write) * d threads;
    }   
}


torch::Tensor softmax_tiled(torch::Tensor V) {
    const int TILE_SIZE = 32;       // Define the tile size
    CHECK_INPUT(V);
    const int d = V.size(0);

    torch::Tensor O = torch::zeros({d}, V.options());

    dim3 tbp(TILE_SIZE);
    dim3 blocks(cdiv(d, TILE_SIZE));
    softmax_tiled_kernel<<<blocks, tbp>>>(
        V.data_ptr<float>(), 
        O.data_ptr<float>(),
        d
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
