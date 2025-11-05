#include <iostream>
#include <cuda.h>

__global__ void ThreadSumReductionKernel(
        float* input, 
        float* output
    ) {
    atomicAdd(output, input[threadIdx.x+10]);

}

int main() {
    // Size of the input data
    const int size = 5;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    int n_threads = 5;  // number of threads in each block = 1024
    int n_blocks = 1;  // Each block is a collection of threads
    ThreadSumReductionKernel<<<n_blocks, n_threads>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "\nExpected sum: " << size << "\n" << std::endl;
    std::cout << "Sum is " << *h_output << "\n" << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
