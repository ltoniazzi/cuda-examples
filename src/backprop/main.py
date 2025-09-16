import torch
import torch.nn as nn
from torch.nn import functional as F
from backprop.ops import SoftMax, MatMul
from backprop.loss import Loss

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
W = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

y = torch.matmul(x, W)
y.retain_grad()

L = F.mse_loss(y, torch.tensor([[1.0, 1.0]]))

L.backward()
x_grad_torch = x.grad
print("x.grad:", x.grad)
print("W.grad:", W.grad)
print("y.grad:", y.grad)



    
matmul = MatMul(weight=W)
y_ = matmul.forward(input=x)
print(y_)

# L = F.mse_loss(y, torch.tensor([[1.0, 1.0]]))
grad_output_torch = y.grad
print(grad_output_torch)
print("grad_output_torch ", grad_output_torch.size())
print("W ", W.size())
print("x ", x.size())

d_weights, grad_output = matmul.backward(grad_output=grad_output_torch)
print("d_weights ", d_weights)
print("grad_output ", grad_output)
assert torch.equal(x_grad_torch, grad_output)
assert torch.equal(W.grad, d_weights)
 


class TorchAttentionNN(nn.Module):
    def __init__(self, W1, W2) -> None:
        super(TorchAttentionNN, self).__init__()
        
        # Define layers
        self.layer_1 = nn.Linear(W1.size(0), W1.size(1), bias=False)
        self.layer_2 = nn.Linear(W2.size(0), W2.size(1), bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        with torch.no_grad():
            self.layer_1.weight.copy_(W1.T)
            self.layer_2.weight.copy_(W2.T)

    def forward(self, x):
        print("Torch")
        x = self.layer_1(x)  # First linear layer
        print(x[:3])
        x = self.softmax(x)  # Softmax activation
        print(x[:3])
        x = self.layer_2(x)  # Second linear layer
        print(x[:3])
        return x.squeeze(-1)



class CustomAttentionNN:
    """An Attention network"""

    def __init__(self, W1, W2) -> None:
        
        self.layer_1 = MatMul(weight=W1)
        self.layer_2 = MatMul(weight=W2)
        self.act = SoftMax()

        self.graph = [
            (self.layer_1, self.act),
            (self.act, self.layer_2),
            (self.layer_2, "END"),
        ]


    def forward(self, x):
        print("Custom")
        x = self.layer_1.forward(x)
        print(x[:3])
        x = self.act.forward(x)
        print(x[:3])
        x = self.layer_2.forward(x)
        print(x[:3])
        return x.squeeze(-1)
    



# Define the dimensions
N, M = 4, 8

# Initialize weights and inputs
W1 = torch.rand((N, M), requires_grad=True)
W2 = torch.rand((M, 1), requires_grad=True)
x = torch.rand((N, N), requires_grad=True)
y_true = torch.rand((N), requires_grad=False)

# Clone weights for the PyTorch model
W1_clone = W1.clone().detach().requires_grad_(True)
W2_clone = W2.clone().detach().requires_grad_(True)
x_clone = x.clone().detach().requires_grad_(True)

# Custom model
model = CustomAttentionNN(W1, W2)
L = Loss(model)
y = model.forward(x)
loss = L.mse_loss(y, y_true)
L.backward()

# Torch model
model_torch = TorchAttentionNN(W1_clone, W2_clone)
y_torch = model_torch(x_clone)
L_torch = F.mse_loss(y_torch, y_true)
y_torch.retain_grad()
L_torch.backward()

print(f"{y_torch.grad=}")

# Assert that the losses are approximately equal
assert torch.allclose(y, y_torch, atol=1e-6), "Losses are not equal"
assert torch.allclose(loss, L_torch, atol=1e-6), "Losses are not equal"

# Assert that the gradients are approximately equal
assert torch.allclose(W2.grad.T, model_torch.layer_2.weight.grad, atol=1e-6), "Gradients for W2 are not equal"
assert torch.allclose(W1.grad.T, model_torch.layer_1.weight.grad, atol=1e-6), "Gradients for W1 are not equal"
assert torch.allclose(x.grad, x_clone.grad, atol=1e-6), "Gradients for x are not equal"