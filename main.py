import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))

dtype = torch.float
device = torch.device("cuda:0")

data_set = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)


class Cross_Correlational_Conceptor:

    def __init__(self):
        print("init")
        self.weights = []
        self.new_weights = []
        self.depth = 0
        self.kernel_size = (3, 3)
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.stride = (1, 1)

    def expand(self, expand_depth, include_depth):
        print("expand")
        A = torch.randn(expand_depth, include_depth, self.kernel_size[0], self.kernel_size[1], requires_grad=True, device=device)

        self.weights.append(A)
        self.new_weights.append(A)

    def learn(self, input, expand_depth):
        print("learn")
        self.expand(expand_depth, input.shape[1])

        optimizer = torch.optim.Adam(self.new_weights, lr=0.001)
        criterion = torch.nn.MSELoss(reduction='mean')

        for i in range(20000):

            hidden = self << input
            input_ = self >> hidden
            loss = criterion(input_, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("step:", i, "th, loss:", loss.item())

        print("final loss:", loss.item())
        self.new_weights.clear()

    def __lshift__(self, input):
        res = torch.cat([
            torch.nn.functional.conv2d(input[:, 0:f.shape[1], ...], f, stride=self.stride, padding=self.padding)
            for f in self.weights
        ], dim=1)
        return res

    # https://github.com/vdumoulin/conv_arithmetic
    def __rshift__(self, hidden):

        h_out = (hidden.shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        w_out = (hidden.shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        depth_out = 0
        for f in self.weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([hidden.shape[0], depth_out, h_out, w_out], device=device)

        from_depth = 0
        for f in self.weights:
            occupied_depth = f.shape[0]
            to_depth = from_depth + occupied_depth
            addition = torch.nn.functional.conv_transpose2d(hidden[:, from_depth:to_depth, ...], f, stride=self.stride, padding=self.padding)
            canvas[:, 0:occupied_depth, ...] = canvas[:, 0:occupied_depth, ...] + addition
            from_depth = to_depth

        return canvas


if __name__ == "__main__":
    print("main")

    layer = Cross_Correlational_Conceptor()
    criterion = torch.nn.MSELoss(reduction='mean')

    input_0 = data_set[0][0].unsqueeze(1).to(device)

    for i, (data, label) in enumerate(data_loader):

        input = data.to(device)
        layer.learn(input, 5)

        gen_0 = layer >> (layer << input_0)
        loss_0 = criterion(gen_0, input_0)
        print(loss_0.item())

        if i == 9:
            break
