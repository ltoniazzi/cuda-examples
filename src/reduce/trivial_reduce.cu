#include <iostream>
#include <cuda.h>

__global__ void TrivialSumReductionKernel(
        float* input, 
        float* output, 
        int input_size
    ) {
    float sum = 0.0f;
    for (int pos = 0; pos < input_size; pos+=1) {
        sum += input[pos];  // sum iteratively
    }
    output[0] = sum;  // Write sum to output
}

int main() {
    // Size of the input data
    const int size = 2048;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* host_input = new float[size];
    float* host_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        host_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* device_input;
    float* device_output;
    cudaMalloc(&device_input, bytes);
    cudaMalloc(&device_output, sizeof(float));

    // Copy data from host to device.  ~ .to(device) in python
    cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice);


    // Up to this point
    // input = torch.Tensor([1.0]*size).to("cuda")


    // Set number of blocks and threads (per block)
    int n_blocks = 1; int n_threads = 1; 

    // Launch the kernel
    TrivialSumReductionKernel<<<n_blocks, n_threads>>>(device_input, device_output, size);

    std::cout << "Array size: " << size << std::endl;
    std::cout << "Blocks: " << n_blocks << ", Threads per block: " << n_threads << std::endl;
    std::cout << "\nExpected result: " << size << std::endl;
    // Copy result back to host
    cudaMemcpy(host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Actual result:   " << *host_output << "\n" << std::endl;

    // Cleanup
    delete[] host_input;
    delete host_output;
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
