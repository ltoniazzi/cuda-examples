import torch
from torch.nn import functional as F

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

    def __init__(self):
        self.input = None # N
        self.weight = None # M, N

    def forward(self, input, weight):
        self.input = input
        self.weight = weight
        return input @ weight

    def backward(self, grad_output): # M
        # Compute dL/dW = grad_output * input
        d_weights =  self.input.T @ grad_output # M, 1 * (1, N) = M, N

        # compute dL/dinput to pass to next step 
        grad_output = grad_output @ self.weight.T

        # Free resources
        return d_weights, grad_output
    
matmul = MatMul()
y_ = matmul.forward(input=x, weight=W)
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