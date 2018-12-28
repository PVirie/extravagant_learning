import torch
import torchvision
import numpy as np


class Cross_Correlational_Conceptor:

    def __init__(self, device):
        print("init")
        self.device = device
        self.weights = []
        self.new_weights = []
        self.depth = 0
        self.kernel_size = (3, 3)
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.stride = (1, 1)

    def learn(self, input, expand_depth):
        print("learn")

        if len(self.weights) is not 0:
            hidden = self << input
            input_ = self >> hidden
        else:
            input_ = 0

        # expand
        A = torch.randn(expand_depth, input.shape[1], self.kernel_size[0], self.kernel_size[1], requires_grad=True, device=self.device)
        self.new_weights.append(A)

        optimizer = torch.optim.Adam(self.new_weights, lr=0.01)
        criterion = torch.nn.MSELoss(reduction='mean')

        with torch.no_grad():
            residue = input - input_

        for i in range(10000):

            new_hidden = self.__internal__forward(input, self.new_weights)
            residue_ = self.__internal__backward(new_hidden, self.new_weights)

            loss = criterion(residue_, residue)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("step:", i, "th, loss:", loss.item())

        print("final loss:", loss.item())

        # merge
        self.weights.append(A)
        self.new_weights.clear()

    def __lshift__(self, input):
        res = self.__internal__forward(input, self.weights)
        return res

    def __internal__forward(self, input, weights):
        res = torch.cat([
            torch.nn.functional.conv2d(input[:, 0:f.shape[1], ...], f, stride=self.stride, padding=self.padding)
            for f in weights
        ], dim=1)
        return res

    # https://github.com/vdumoulin/conv_arithmetic
    def __rshift__(self, hidden):
        canvas = self.__internal__backward(hidden, self.weights)
        return canvas

    def __internal__backward(self, hidden, weights):

        h_out = (hidden.shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        w_out = (hidden.shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        depth_out = 0
        for f in weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([hidden.shape[0], depth_out, h_out, w_out], device=self.device)

        from_depth = 0
        for f in weights:
            occupied_depth = f.shape[0]
            to_depth = from_depth + occupied_depth
            addition = torch.nn.functional.conv_transpose2d(hidden[:, from_depth:to_depth, ...], f, stride=self.stride, padding=self.padding)
            canvas[:, 0:occupied_depth, ...] = canvas[:, 0:occupied_depth, ...] + addition
            from_depth = to_depth

        return canvas


if __name__ == '__main__':
    print("conceptor")
