🧠 TL;DR
Layer	What it does	Link
Python API	torch.nn.functional.softmax()	functional.py
C++ dispatch	ATen/native/SoftMax.cpp	SoftMax.cpp
CUDA backend	ATen/native/cuda/SoftMax.cu	SoftMax.cu
Triton/Flash	Fused/traced	[Triton softmax tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py)

Would you like help reproducing PyTorch’s softmax in Triton or reimplementing it exactly like their CUDA version?