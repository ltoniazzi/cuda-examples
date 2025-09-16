import torch

class Loss:

    def __init__(self, model) -> None:
        self.loss = None
        self.loss_grad = None
        self.model = model
        
    def mse_loss(self, y, y_true):

        # y = y.squeeze(-1)
        self.loss = torch.mean(
            torch.square(y-y_true)
        )
        n = y.numel()
        self.loss_grad = (2/n)*(y-y_true)
        return self.loss
    
    def backward(self):
        dL_dy = self.loss_grad
        print(f"{dL_dy=}")
        dy_dW2, grad_output_W2 = self.model.layer_2.backward(dL_dy.unsqueeze(-1))
        d_softmax = self.model.act.backward(grad_output_W2)
        dS_dW1, grad_output_W1 = self.model.layer_1.backward(d_softmax)
        return dS_dW1

