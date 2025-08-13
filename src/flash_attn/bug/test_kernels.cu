#include <cuda_runtime.h>
#include <iostream>

__global__ void test_not_works(float* O, int d, int factor_by, int factored_d) {
    int tid_x = threadIdx.x;
    
    // Local array (private to each thread)
    float O_local[2];  // factored_d = 2
    
    // Initialize local array
    for (int i = 0; i < factored_d; i++) {
        O_local[i] = 0.0f;
    }
    
    // First loop: accumulate
    for (int dd = tid_x; dd < d; dd += blockDim.x) {
        O_local[dd / factor_by] += 1.0f;
    }
    
    // Second loop: write to global memory
    for (int dd = tid_x; dd < d; dd += blockDim.x) {
        O[dd] = O_local[dd / factor_by];
    }
}

__global__ void test_works(float* O, int d) {
    int tid_x = threadIdx.x;
    
    // Local array (private to each thread)
    float O_local[4];  // d = 4
    
    // Initialize local array
    for (int i = 0; i < d; i++) {
        O_local[i] = 0.0f;
    }
    
    // First loop: accumulate
    for (int dd = tid_x; dd < d; dd += blockDim.x) {
        O_local[dd] += 1.0f;
    }
    
    // Second loop: write to global memory
    for (int dd = tid_x; dd < d; dd += blockDim.x) {
        O[dd] = O_local[dd];
    }
}

int main() {
    const int d = 4;
    const int block_dim_x = 2;
    const int block_dim_y = 1;
    const int grid_size_y = 1;
    const int factor_by = block_dim_x;
    const int factored_d = d / factor_by;
    
    // Allocate host memory
    float* h_O_not_works = new float[d];
    float* h_O_works = new float[d];
    
    // Allocate device memory
    float* d_O_not_works;
    float* d_O_works;
    cudaMalloc(&d_O_not_works, d * sizeof(float));
    cudaMalloc(&d_O_works, d * sizeof(float));
    
    // Initialize device memory to zero
    cudaMemset(d_O_not_works, 0, d * sizeof(float));
    cudaMemset(d_O_works, 0, d * sizeof(float));
    
    // Configure kernel launch parameters
    dim3 blocks(block_dim_y, block_dim_x);  // (1, 2)
    dim3 grid(grid_size_y);                 // (1)
    
    // Launch kernels
    test_not_works<<<grid, blocks>>>(d_O_not_works, d, factor_by, factored_d);
    cudaDeviceSynchronize();
    
    test_works<<<grid, blocks>>>(d_O_works, d);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_O_not_works, d_O_not_works, d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_works, d_O_works, d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "O_not_works: [";
    for (int i = 0; i < d; i++) {
        std::cout << h_O_not_works[i];
        if (i < d-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "O_works: [";
    for (int i = 0; i < d; i++) {
        std::cout << h_O_works[i];
        if (i < d-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Cleanup
    delete[] h_O_not_works;
    delete[] h_O_works;
    cudaFree(d_O_not_works);
    cudaFree(d_O_works);
    
    return 0;
}
