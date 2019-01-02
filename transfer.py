import torch
import torchvision


def interleave(seq, dim=1):
    stacked = torch.stack(seq, dim=dim + 1)
    out_shape = list(seq[0].shape)
    out_shape[dim] = -1
    return torch.reshape(stacked, out_shape)


def invert_interleave(input, dim=1, chunks=2):
    expanded_shape = list(input.shape)
    expanded_shape[dim] = -1
    expanded_shape.insert(dim + 1, chunks)
    expanded = torch.reshape(input, expanded_shape)
    chunks = torch.unbind(expanded, dim=dim + 1)
    return chunks


class Mirroring_Relu_Layer:

    def __init__(self, device):
        print("init")

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            p = torch.nn.functional.relu(input)
            n = torch.nn.functional.relu(-input)
            res = interleave([p, n], dim=1)
        return res

    def __rshift__(self, hidden):
        with torch.no_grad():
            p, n = invert_interleave(hidden, dim=1, chunks=2)
            res = p - n
        return res


if __name__ == '__main__':
    print("assert mirroring relu preserves the containment property.")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Mirroring_Relu_Layer(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    x = torch.randn(1, 5, 28, 28, device=device)
    x2 = torch.randn(1, 5, 28, 28, device=device)
    xs = torch.cat([x, x2], dim=1)

    hidden = layer << xs
    print(hidden.shape)
    xs_ = layer >> hidden

    loss = criterion(xs_[:, 0:5, ...], x)
    print(loss.item())
