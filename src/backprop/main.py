import torch
from torch.nn import functional as F
import os
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported


x = torch.tensor([[1.0, 2.0]], requires_grad=True)
W = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

y = torch.matmul(x, W)
y.retain_grad()

L = F.mse_loss(y, torch.tensor([[1.0, 1.0]]))

L.backward()

print("x.grad:", x.grad)
print("W.grad:", W.grad)
print("y.grad:", y.grad)



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
    
matmul = MatMul(weight=W)
y_ = matmul.forward(input=x)
print(y_)

# L = F.mse_loss(y, torch.tensor([[1.0, 1.0]]))
grad_output = y.grad
print(grad_output)
print("grad_output ", grad_output.size())
print("W ", W.size())
print("x ", x.size())

d_weights, grad_output = matmul.backward(grad_output=grad_output)
print("d_weights ", d_weights)
print("grad_output ", grad_output)
    
# class MatMul(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input, weight)
#         return input @ weight

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         grad_input = grad_output @ weight.t()
#         grad_weight = input.t() @ grad_output
#         return grad_input, grad_weight


# Tritol kernel for softmax
import triton 
import triton.language as tl


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

M = torch.Tensor([
    [1.0, 2.0],
    [3.0, 5.0],
])
S = softmax(M)
print("S ", S)

S_torch = F.softmax(M, dim=-1)
print("S_torch ", S_torch)


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
    

class NN:
    """An Attention network"""

    def __init__(self, N, M) -> None:
        W1 = torch.rand((N, M))
        W2 = torch.rand((M, 1))

        self.layer_1 = MatMul(weight=W1)
        self.layer_2 = MatMul(weight=W2)
        self.act = SoftMax()

        self.graph = [
            (self.layer_1, self.act),
            (self.act, self.layer_2),
            (self.layer_2, "END"),
        ]


    def forward(self, x):
        x = self.layer_1.forward(x)
        x = self.act.forward(x)
        x = self.layer_2.forward(x)
        return x.squeeze(-1)
    

N, M = 4, 5

model = NN(4, 5)
x = torch.rand((N, N))
y = model.forward(x)
y_true = torch.rand((N))

class Loss:

    def __init__(self, model) -> None:
        self.loss = None
        self.loss_grad = None
        self.model = model
        
    def mse_loss(self, y, y_true):

        self.loss = torch.mean(
            torch.square(y-y_true)
        )
        self.loss_grad = 2*(y-y_true)
        return self.loss
    
    def backward(self):
        dL_dy = self.loss_grad
        dy_dW2, grad_output_W2 = self.model.layer_2.backward(dL_dy)
        d_softmax = self.model.act.backward(grad_output_W2)
        dS_dW1, grad_output_W1 = self.model.layer_1.backward(d_softmax)
        return dS_dW1


L = Loss(model)

loss = L.mse_loss(y, y_true)
L_torch = F.mse_loss(y, y_true)

assert L_torch == loss


L.backward()