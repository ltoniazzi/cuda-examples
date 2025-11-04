#include <iostream>
#include <cuda.h>


# define MAX_THREADS 1024

__global__ void BlockSumReductionKernel(float* input, float* output) {
    // Super-fast but super small memory shared in the block
    __shared__ float input_shared[MAX_THREADS];  
    int thread_id = threadIdx.x;
    int block_dim = blockDim.x;
    int block_id = blockIdx.x;


    int global_thread_id = thread_id + block_id * block_dim;
    int thread_write_location = 2 * global_thread_id;
    

    // Do first addition between the mapped threads spaced by 2 and the next value
    input_shared[thread_id] = input[thread_write_location] + input[thread_write_location + 1];

    // Now all alememnts have been read from HBM/Device memory and live in shared
    __syncthreads();

    // Now reduce summing with the new input_shared of size 1024, start from stride=1  to block_dim/2=512
    for (int stride = 1; stride <= block_dim/2; stride *= 2) {
        if ( thread_id % (2 * stride) == 0 ) {
            input_shared[thread_id] += input_shared[thread_id + stride];
        }
         __syncthreads();
    }

    // The block is complete, so let's add to output

    //Reduce over blocks summing the first element of the block's input_shared
    if (thread_id == 0){
        atomicAdd(output, input_shared[0]);
    }
}

int main() {
    // Size of the input data
    const int size = 2048*250000;
    const int bytes = size * sizeof(float);

    // Work only with size multiple of 2048

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
    cudaMemset(d_output, 0, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    int n_threads = MAX_THREADS;
    int elements_a_block_can_process = MAX_THREADS*2;
    int n_blocks = size / elements_a_block_can_process;


    std::cout << "Array size: " << size << std::endl;
    std::cout << "Blocks: " << n_blocks << ", Threads per block: " << n_threads << std::endl;
    std::cout << "Elements processed: " << n_blocks * n_threads * 2 << std::endl;
    std::cout << "\nExpected result: " << size << std::endl;


    BlockSumReductionKernel<<<n_blocks, n_threads>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "\nSum is " << *h_output << "\n" << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
