# Cuda Examples
Collection of example kernels:

- [Tiled MatMul](src/matmul): A simple implementation of tiled multiplication
- [1D Softmax](src/softmax_1d): different implementations of 1D softmax with some profiling
- [Flash Atetntion](src/flash_attn): Implementation of fused matmul and softmax and then flash attention.


## Setup

```shell
make setup
```


Tested on: 
- NVIDIA A10G
- CUDA Version: 12.6