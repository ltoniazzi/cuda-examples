# Tritol kernel for softmax
import os
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import triton 
import triton.language as tl
import torch

def cdiv(a, b):
    return (a+b-1) // b

@triton.jit
def softmax_kernel(M_ptr, S_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr):
    #each block does 1 row  
    pid = tl.program_id(0)
    if pid < n_rows:

        offs = pid*n_cols + tl.arange(0, n_rows)

        row = tl.load(M_ptr + offs)

        row = tl.exp(row)

        l = tl.sum(row)

        row = row / l

        tl.store(S_ptr+ offs, row)


def softmax(M):
    S = torch.zeros_like(M)
    n_rows, n_cols = M.size()
    n_blocks = n_rows
    grid = (n_blocks,)  # how many blocks do we have? can be 1d/2d/3d-tuple or function returning 1d/2d/3d-tuple

    # launch grid!
    # - kernel_fn is the triton kernel, which we write below
    # - grid is the grid we constructed above
    # - x,z,n,bs are paramters that are passed into each kernel function
    softmax_kernel[grid](M,S,n_rows, n_cols)

    return S  


class SoftMax:
    def __init__(self):
        self.input = None # M, N
        self.output = None # M, N

    def forward(self, input):
        self.input = input
        self.output = softmax(input)
        return self.output

    def backward(self, grad_output): # M
        # Compute dL/dW = grad_output * input
        d_softmax = torch.mul(self.output, (torch.ones_like(self.output) - self.output )) # M, 1 * (1, N) = M, N
        return d_softmax
    

class MatMul:

    def __init__(self, weight):
        self.input = None # N
        self.weight = weight # M, N

    def forward(self, input):
        self.input = input
        return input @ self.weight

    def backward(self, grad_output): # M
        # Compute dL/dW = grad_output * input
        d_weights =  self.input.T @ grad_output # M, 1 * (1, N) = M, N
        self.weight.grad = d_weights.expand_as(self.weight)

        # compute dL/dinput to pass to next step 
        grad_output = grad_output @ self.weight.T

        # Free resources
        return d_weights, grad_output
    

if __name__ == "__main__":
    
    M = torch.Tensor([
        [1.0, 2.0],
        [3.0, 5.0],
    ])
    S = softmax(M)
    print("S ", S)

    S_torch = torch.nn.functional.softmax(M, dim=-1)
    print("S_torch ", S_torch)
