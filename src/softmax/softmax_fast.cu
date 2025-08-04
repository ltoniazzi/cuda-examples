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
__global__ void softmax_kernel(float* input, float* output, int d) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Guard against out-of-bounds thread
    float val = (tid < d) ? input[tid] : -INFINITY;
    if (tid < d) sdata[tid] = val;
    __syncthreads();

    // Step 1: find max
    float max_val = -INFINITY;
    for (int i = 0; i < d; i++) {
        max_val = fmaxf(max_val, sdata[i]);
    }
    __syncthreads();

    // Step 2: compute exp(x - max)
    float exp_val = 0.0f;
    if (tid < d) {
        exp_val = expf(sdata[tid] - max_val);
        sdata[tid] = exp_val;
    }
    __syncthreads();

    // Step 3: compute sum
    float sum_val = 0.0f;
    for (int i = 0; i < d; i++) {
        sum_val += sdata[i];
    }
    __syncthreads();

    // Step 4: normalize
    if (tid < d) {
        output[tid] = exp_val / sum_val;
    }
}


torch::Tensor softmax(torch::Tensor V) {
    CHECK_INPUT(V);
    TORCH_CHECK(V.dim() == 1, "Input must be a 1D tensor");

    const int d = V.size(0);
    torch::Tensor O = torch::empty_like(V);

    const int maxThreads = 1024;
    const int threads = std::min(d, maxThreads);
    const int shared_mem_bytes = threads * sizeof(float);

    softmax_kernel<<<1, threads, shared_mem_bytes>>>(
        V.data_ptr<float>(), 
        O.data_ptr<float>(), 
        d
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
