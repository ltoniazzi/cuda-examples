# Some examples of building cuda kernels for 1D softmax

In this folder we have:
- "fast" implementation (for 1D tensors that fit in a block)
- Tiled implementaion (slower than fast)
- Naive and unsafe implmentation
- Triton implememntation from the triton examples.

See notebook for profiling example of the custom kernels.
Run triton script fot its benchmarking
 

## Real implementaion
| Layer           | What it does                 | Link                                                                                      |
|----------------|------------------------------|-------------------------------------------------------------------------------------------|
| Python API     | `torch.nn.functional.softmax()` | `functional.py`                                                                          |
| C++ dispatch   | `ATen/native/SoftMax.cpp`     | `SoftMax.cpp`                                                                             |
| CUDA backend   | `ATen/native/cuda/SoftMax.cu` | [SoftMax.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu) |
| Triton/Flash   | Fused/traced                  | [Triton softmax tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py) |



### Try see PTX:
To see the PTX (Parallel Thread Execution) code generated from your CUDA kernel, you have a few options depending on your goal:

✅ 1. Compile with --ptx using nvcc
This is the simplest and most direct method.

🔧 Command:
```bash
nvcc -ptx your_kernel.cu -o your_kernel.ptx
```
📄 What you get:
A .ptx file containing the human-readable assembly used by NVIDIA GPUs.

✅ 2. Use --keep to inspect all intermediate files
This is useful if you're compiling .cu to binary (.cubin, .fatbin, etc.) but still want to see PTX.

🔧 Command:
```bash
nvcc --keep --keep-dir ./tmp -O3 your_kernel.cu
```
This will produce:

.ptx files

.cubin files

.o object files

In the ./tmp directory.

✅ 3. Show register usage from ptxas
If you want to see register pressure (important for performance tuning):

🔧 Command:
```bash
nvcc -O3 --ptxas-options=-v your_kernel.cu -o your_kernel.o
```
📋 Output:
```nginx
ptxas info    : Compiling entry function '_Z18softmax_kernel...' for 'sm_86'
ptxas info    : Used 20 registers, 96 bytes smem, 8 bytes cmem[0]
```
✅ Use this to monitor register count, which directly affects occupancy and performance.

✅ 4. Disassemble a compiled binary
If you have a .cubin or .fatbin (e.g. from PyTorch extensions), use:

🔧 Command:
```bash
nvdisasm your_kernel.fatbin > disassembled.sass
```
This gives you SASS, the actual GPU ISA (lower level than PTX).

🧠 Bonus: From PyTorch Custom CUDA Extension
If you're building a CUDA extension with PyTorch (load() or load_inline()), and want to extract the PTX:

The compiled .cu file goes to ~/.cache/torch_extensions/your_extension/

Navigate there and run:

```bash
nvcc -ptx cuda.cu -o cuda.ptx
```
Or:

```bash
nvcc --ptxas-options=-v cuda.cu -o cuda.o
```