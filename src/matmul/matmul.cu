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


// MatMul kernel
__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int h, int w, int k) {
    const int TILE_SIZE = 16; // Define the tile size
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Identify thread and map to matrix indices
    int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x*blockDim.x + threadIdx.x; 
    int rowIdxTile = threadIdx.y;
    int colIdxTile = threadIdx.x; 


    int nTiles = cdiv(k, TILE_SIZE);

    float res = 0.0f;

    for (int nTile=0; nTile < nTiles; nTile++) {
        if (rowIdx < h && nTile * TILE_SIZE + colIdxTile < k) 
            As[rowIdxTile][colIdxTile] = A[
                rowIdx * k                         // Go to the right row
                + nTile * TILE_SIZE + colIdxTile    // Iterate on the respecctive tile element in each tile
            ];
        else
            As[rowIdxTile][colIdxTile] = 0.0f;

        if (nTile * TILE_SIZE + rowIdxTile < k && colIdx < w)
            Bs[rowIdxTile][colIdxTile] = B[
                w * (
                    nTile*TILE_SIZE   // number of rows to skip for the previuous tiles
                    + rowIdxTile    // number of rows to skip for the current tile
                ) 
                + colIdx                        // Go to the right column
            ];
        else
            Bs[rowIdxTile][colIdxTile] = 0.0f;
        __syncthreads();
        
        for (int tile_k=0; tile_k< TILE_SIZE; tile_k++) {
            res += As[rowIdxTile][tile_k]*Bs[tile_k][colIdxTile];
        };
        __syncthreads();
    };
    if (rowIdx < h && colIdx < w) {
        C[w*rowIdx + colIdx] = res;
    };
}


torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    const int TILE_SIZE = 16; // Define the tile size
    CHECK_INPUT(A); CHECK_INPUT(B);
    const int h = A.size(0);
    const int k = A.size(1);
    const int w = B.size(1);

    assert (A.size(1) == B.size(0) && "Matrix dimensions do not match for multiplication");
    torch::Tensor C = torch::zeros({h, w}, A.options());


    dim3 tbp(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(w, TILE_SIZE), cdiv(h, TILE_SIZE));
    matmul_tiled_kernel<<<blocks, tbp>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        h, 
        w,
        k
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
