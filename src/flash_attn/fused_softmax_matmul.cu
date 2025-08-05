#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuQssert((ans), __FILE__, __LINE__); }
inline void gpuQssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}


// MatMul kernel
__global__ void fused_softmax_matmul_kernel(float* Q, float* K, float* O, int h, int w, int k) {
    const int TILE_SIZE = 32; // Define the tile size
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];

    // Identify thread and map to matrix indices
    int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x*blockDim.x + threadIdx.x; 
    int rowIdxTile = threadIdx.y;
    int colIdxTile = threadIdx.x; 


    int nTilesHors = cdiv(w, TILE_SIZE);
    int nTileInns = cdiv(k, TILE_SIZE);

    float denominator = 0.0f;
    float numerator_exponent = 0.0f;
    float max_new = -INFINITY;
    float max_prev = -INFINITY;

    for (int nTilesHor=0; nTilesHor < nTilesHors; nTilesHor++) {
        float res_inn = 0.0f;

        for (int nTileInn=0; nTileInn < nTileInns; nTileInn++) {
            if (rowIdx < h && nTileInn * TILE_SIZE + colIdxTile < k) 
                Qs[rowIdxTile][colIdxTile] = Q[
                    rowIdx * k                         // Go to the right row
                    + nTileInn * TILE_SIZE + colIdxTile    // Iterate on the respecctive tile element in each tile
                ];
            else
                Qs[rowIdxTile][colIdxTile] = 0.0f;

            if (nTileInn * TILE_SIZE + rowIdxTile < k && nTilesHor*TILE_SIZE + colIdxTile < w)
                Ks[rowIdxTile][colIdxTile] = K[
                    w * (
                        nTileInn*TILE_SIZE   // number of rows to skip for the previuous tiles
                        + rowIdxTile    // number of rows to skip for the current tile
                    ) 
                    + nTilesHor*TILE_SIZE + colIdxTile                        // Go to the right column scanning horizontally
                ];
            else
                Ks[rowIdxTile][colIdxTile] = 0.0f;
            __syncthreads();
            
            for (int tile_k=0; tile_k< TILE_SIZE; tile_k++) {
                res_inn += Qs[rowIdxTile][tile_k] * Ks[tile_k][colIdxTile];
            };
            if (nTilesHor*TILE_SIZE + colIdxTile == colIdx) {
                numerator_exponent = res_inn;
            }
            __syncthreads();
        }
    
        max_new = fmaxf(max_prev, res_inn);
        if (max_new == max_prev) {
            denominator += __expf(res_inn - max_new);
        }
        else {
            denominator = denominator * __expf(max_prev - max_new) + 1.0f;
            max_new = max_prev;
        }

    }

    if (rowIdx < h && colIdx < w) {
        O[w*rowIdx + colIdx] = __expf(numerator_exponent - max_new)/denominator;
    };

}


torch::Tensor fused_softmax_matmul(torch::Tensor Q, torch::Tensor K) {
    const int TILE_SIZE = 32; // Define the tile size
    CHECK_INPUT(Q); CHECK_INPUT(K);
    const int h = Q.size(0);
    const int k = Q.size(1);
    const int w = K.size(1);

    assert (Q.size(1) == K.size(0) && "Matrix dimensions do not match for multiplication");
    torch::Tensor O = torch::zeros({h, w}, Q.options());


    dim3 tbp(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(w, TILE_SIZE), cdiv(h, TILE_SIZE));
    fused_softmax_matmul_kernel<<<blocks, tbp>>>(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        O.data_ptr<float>(), 
        h, 
        w,
        k
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
