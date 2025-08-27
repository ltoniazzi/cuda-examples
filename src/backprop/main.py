import torch
from torch.nn import functional as F
from backprop.ops import SoftMax, MatMul
from backprop.loss import Loss

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
W = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

y = torch.matmul(x, W)
y.retain_grad()

L = F.mse_loss(y, torch.tensor([[1.0, 1.0]]))

L.backward()

print("x.grad:", x.grad)
print("W.grad:", W.grad)
print("y.grad:", y.grad)



    
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
 






class NN:
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
        x = self.layer_1.forward(x)
        x = self.act.forward(x)
        x = self.layer_2.forward(x)
        return x.squeeze(-1)
    

N, M = 4, 5
W1 = torch.rand((N, M))
W2 = torch.rand((M, 1))
model = NN(W1, W2)
x = torch.rand((N, N))
y = model.forward(x)
y_true = torch.rand((N))


L = Loss(model)

loss = L.mse_loss(y, y_true)
L_torch = F.mse_loss(y, y_true)

assert L_torch == loss


L.backward()