import torch


def sum_norm(A):
    return torch.sum(torch.norm(torch.reshape(A, [A.shape[0], -1]), dim=1))


class Layer:
    def __init__(self):
        print("do nothing")

    def learn(self, input, expand_depth, expand_threshold=1e-6, steps=1000, lr=0.01):
        print("do nothing")
