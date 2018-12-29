import torch
import torchvision


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

    def learn(self, input, expand_depth, steps=1000):
        print("learn")

        if len(self.weights) is not 0:
            hidden = self.__internal__forward(input, self.weights)
            input_ = self.__internal__backward(hidden, self.weights, input.shape[1])
        else:
            input_ = torch.zeros(1, input.shape[1], 1, 1, device=self.device)

        # expand
        A = torch.empty(expand_depth, input.shape[1], self.kernel_size[0], self.kernel_size[1], device=self.device, requires_grad=True)
        torch.nn.init.normal_(A, 0, 0.001)
        self.new_weights.append(A)

        optimizer = torch.optim.Adam(self.new_weights, lr=0.01)
        criterion = torch.nn.MSELoss(reduction='mean')

        with torch.no_grad():
            residue = input - input_

        for i in range(steps):

            new_hidden = self.__internal__forward(input, self.new_weights)
            residue_ = self.__internal__backward(new_hidden, self.new_weights)

            loss = criterion(residue_, residue)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("step:", i, "th, loss:", loss.item())

        print("final loss:", loss.item())

        # merge
        self.weights.append(A)
        self.new_weights.clear()

    def __internal__forward(self, input, weights):
        res = torch.cat([
            torch.nn.functional.conv2d(input[:, 0:f.shape[1], ...], f, stride=self.stride, padding=self.padding)
            for f in weights
        ], dim=1)
        return res

    # https://github.com/vdumoulin/conv_arithmetic
    def __internal__backward(self, hidden, weights, depth_out=0):

        h_out = (hidden.shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        w_out = (hidden.shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        for f in weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([hidden.shape[0], depth_out, h_out, w_out], device=self.device)

        from_depth = 0
        for f in weights:
            to_depth = from_depth + f.shape[0]
            addition = torch.nn.functional.conv_transpose2d(hidden[:, from_depth:to_depth, ...], f, stride=self.stride, padding=self.padding)
            occupied_depth = f.shape[1]
            canvas[:, 0:occupied_depth, ...] = canvas[:, 0:occupied_depth, ...] + addition
            from_depth = to_depth

        return canvas

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            res = self.__internal__forward(input, self.weights)
        return res

    def __rshift__(self, hidden):
        with torch.no_grad():
            canvas = self.__internal__backward(hidden, self.weights)
        return canvas


if __name__ == '__main__':
    print("test conceptor")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer1 = Cross_Correlational_Conceptor(device)
    layer2 = Cross_Correlational_Conceptor(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    x1 = torch.randn(1, 5, 28, 28, device=device)
    x2 = torch.randn(1, 5, 28, 28, device=device)

    layer1.learn(x1, 10)

    x1_1 = layer1 << x1
    print(x1_1.shape)

    layer2.learn(x1_1, 10)

    x1_2 = layer2 << x1_1
    print(x1_2.shape)
    x_ = layer1 >> (layer2 >> x1_2)

    loss = criterion(x_, x1)
    print(loss.item())

    layer1.learn(x2, 5)

    x2_1 = layer1 << x2
    print(x2_1.shape)

    layer2.learn(x2_1, 5)

    hidden = (layer2 << (layer1 << x1))
    print(hidden.shape)
    x_ = layer1 >> (layer2 >> hidden)

    loss = criterion(x_, x1)
    print(loss.item())
