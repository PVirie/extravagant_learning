import torch
import torchvision


class Mirroring_Relu_Layer:

    def __init__(self, device):
        print("init")

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            p = torch.nn.functional.relu(input)
            n = torch.nn.functional.relu(-input)
            res = torch.cat([p, n], dim=1)
        return res

    def __rshift__(self, hidden):
        with torch.no_grad():
            b = int(hidden.shape[1] / 2)
            p = hidden[:, 0:b, ...]
            n = hidden[:, b:, ...]
            res = p - n
        return res


if __name__ == '__main__':
    print("test mirroring relu")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Mirroring_Relu_Layer(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    x = torch.randn(1, 5, 28, 28, device=device)

    hidden = layer << x
    print(hidden.shape)
    x_ = layer >> hidden

    loss = criterion(x_, x)
    print(loss.item())
