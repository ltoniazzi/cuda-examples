// Include deps for cuda



// MatMul kernel
__global__ void matmul_tiled_sqr_kernel(float* A, float* B, float* C, int h, int w, int k) {
    const int TILE_SIZE = 16; // Define the tile size
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Identify thread and map to matrix indices
    int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x*blockDim.x + threadIdx.x; 
    int rowIdxTile = threadIdx.y;
    int colIdxTile = threadIdx.x; 


    int nTiles = cdiv(w, TILE_SIZE);

    float res = 0.0f;

    for (int nTile=0; nTile < nTiles; nTile++) {
        // M_tile[ir][ic] = (((r < h) && (K_tileidx * TILE_SIZE + ic < k)) ? M[r * k + K_tileidx * TILE_SIZE + ic] : 0.f);
        // N_tile[ir][ic] = ((((K_tileidx * TILE_SIZE + ir) < k) && (c < w)) ? N[(K_tileidx * TILE_SIZE + ir) * w + c] : 0.f);
    
        As[rowIdxTile][colIdxTile] = A[
            rowIdx*w                         // Go to the right row
            + nTile*TILE_SIZE +rowIdxTile    // Iterate on the respecctive tile element in each tile
        ]
        Bs[rowIdxTile][colIdxTile] = B[
            // nTile*TILE_SIZE*w + w*rowIdxTile 
            w*(
                nTile*TILE_SIZE   // number of rows to skip for the previuous tiles
                + rowIdxTile    // number of rows to skip for the current tile
            ) 
            + colIdx                        // Go to the right column
        ];


        __syncthreads();
        for (int tile_k=0; tile_k< TILE_SIZE; i++) {
            res += As[rowIdxTile][tile_k]*Bs[tile_k][colIdxTile]
        }

        __syncthreads();

    }
    C[w*rowIdx + c] = res;
}



int cdiv(int a, int b) {
    return (a + b - 1) / b;
}


// Torch function
#include <torch/extension.h>
#include <cuda.h>

Torch matmul_tiled_sqr(A: torch, B: torch) {
    const int TILE_SIZE = 16; // Define the tile size
    const int h = A.size(0);
    const int k = A.size(1);
    const int w = B.size(1);

    assert (A.size(1) == B.size(0) && "Matrix dimensions do not match for multiplication");
    torch::Tensor C = torch::zeros({h, w}, A.options());


    dim3 tbp = dim3(TILE_SIZE, TILE_SIZE);
    dim3 blocks = dim3(cdiv(w, TILE_SIZE), cdiv(h, TILE_SIZE));
    matmul_tiled_sqr_kernel<<<blocks, tbp>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), h, w);
    cudaError_t err = cudaGetLastError();
}
