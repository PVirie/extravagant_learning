import torch
from layer import *
import os
import itertools


def sum_norm(A):
    return torch.sum(torch.norm(torch.reshape(A, [A.shape[0], -1]), dim=1))


class Cross_Correlational_Conceptor(Layer):

    def __init__(self, device, kernel_size=(3, 3), file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.file_path = file_path

    def save(self):
        if self.file_path:
            torch.save(self.weights, self.file_path)

    def load(self):
        if self.file_path:
            self.weights = torch.load(self.file_path)

    def __internal__assign_output_padding(self, input):
        h = input.shape[2]
        w = input.shape[3]

        self.offsets = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.output_padding = (self.kernel_size[0] - (h % self.kernel_size[0]), self.kernel_size[1] - (w % self.kernel_size[1]))
        padded = torch.nn.functional.pad(input, (0, self.output_padding[1], 0, self.output_padding[0]))
        return padded

    def __internal__perspective(self, input):
        res = torch.cat([
            torch.nn.functional.pad(input, (x, self.kernel_size[1] - x, y, self.kernel_size[0] - y))
            for (y, x) in itertools.product(range(self.kernel_size[0]), range(self.kernel_size[1]))
        ], dim=0)
        return res

    def __internal__pool(self, input):
        shape = [-1, self.kernel_size[0] * self.kernel_size[1], input.shape[1], input.shape[2], input.shape[3]]
        return torch.reshape(input, shape)[0]

    def __internal__revert_output_padding(self, output):
        return output[
            :, :,
            self.offsets[0]:(self.offsets[0] + (output.shape[2] - self.kernel_size[0]) - self.output_padding[0]),
            self.offsets[1]:(self.offsets[1] + (output.shape[3] - self.kernel_size[1]) - self.output_padding[1])
        ]

    def learn(self, input, expand_depth=1, expand_threshold=1e-4, expand_steps=1000, steps=1000, lr=0.01, verbose=False):
        print("learn")

        criterion = torch.nn.MSELoss(reduction='mean')

        input = self.__internal__perspective(input)
        input = self.__internal__assign_output_padding(input)

        prev_loss = 0
        for k in range(expand_steps):

            with torch.no_grad():
                if len(self.weights) is not 0:
                    hidden = self.__internal__forward(input, self.weights)
                    input_ = self.__internal__backward(hidden, self.weights, input.shape[1])
                else:
                    input_ = torch.zeros(1, input.shape[1], 1, 1, device=self.device)

                residue = input - input_

            rloss = criterion(input_, input)
            if rloss.item() < expand_threshold:
                print("Stop expansion after", k * expand_depth, "bases, small reconstruction loss.", rloss.item())
                return True
            if abs(rloss.item() - prev_loss) < 1e-6:
                print("Stop expansion after", k * expand_depth, "bases, small delta error.", rloss.item(), prev_loss)
                # del self.weights[len(self.weights) - k:]
                return False

            # expand
            A = torch.empty(expand_depth, input.shape[1], self.kernel_size[0], self.kernel_size[1], device=self.device, requires_grad=True)
            torch.nn.init.xavier_normal_(A)
            new_weights = [A]

            optimizer = torch.optim.Adam(new_weights, lr=lr)
            for i in range(steps):

                new_hidden = self.__internal__forward(input, new_weights)
                residue_ = self.__internal__backward(new_hidden, new_weights)

                loss = criterion(residue_, residue)

                optimizer.zero_grad()
                loss.backward()
                # print(A.grad.shape)
                optimizer.step()
                if i % 100 == 0:
                    if verbose:
                        print("step:", i, "th, loss:", loss.item())

            if verbose:
                print("final loss:", loss.item())

            norm = sum_norm(A).item()
            if (norm / expand_depth - 1)**2 > expand_threshold:
                print("Failed solution, continue...", norm)
                continue

            # merge
            self.weights.append(A)
            prev_loss = rloss.item()

        return False

    def __internal__forward(self, input, weights):
        res = torch.cat([
            torch.nn.functional.conv2d(input[:, 0:f.shape[1], ...], f, stride=self.stride)
            for f in weights
        ], dim=1)
        return res

    def __internal__get_canvas(self, hidden, weights, depth_out=0):

        h_out = hidden.shape[2] * self.kernel_size[0]
        w_out = hidden.shape[3] * self.kernel_size[1]

        for f in weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([hidden.shape[0], depth_out, h_out, w_out], device=self.device)
        return canvas

    # https://github.com/vdumoulin/conv_arithmetic
    def __internal__backward(self, hidden, weights, depth_out=0):

        canvas = self.__internal__get_canvas(hidden, weights, depth_out)

        from_depth = 0
        for f in weights:
            to_depth = from_depth + f.shape[0]
            addition = torch.nn.functional.conv_transpose2d(
                hidden[:, from_depth:to_depth, ...], f,
                stride=self.stride)
            occupied_depth = f.shape[1]
            canvas[:, 0:occupied_depth, ...] = canvas[:, 0:occupied_depth, ...] + addition
            from_depth = to_depth

        return canvas

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            padded = self.__internal__perspective(input)
            nper = self.__internal__assign_output_padding(padded)
            hidden = self.__internal__forward(nper, self.weights)
            pooled = self.__internal__pool(hidden)
        return pooled

    def __rshift__(self, hidden):
        with torch.no_grad():
            canvas = self.__internal__backward(hidden, self.weights)
            output = self.__internal__revert_output_padding(canvas)

        return output


if __name__ == '__main__':
    print("assert conceptor preserves the containment property")

    dtype = torch.float
    device = torch.device("cuda:0")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    layer1 = Cross_Correlational_Conceptor(device, kernel_size=(3, 3), file_path=os.path.join(dir_path, "weights", "conceptor_layer1.wt"))
    layer2 = Cross_Correlational_Conceptor(device, kernel_size=(3, 3))
    criterion = torch.nn.MSELoss(reduction='mean')

    x1 = torch.randn(1, 5, 28, 28, device=device)
    x2 = torch.randn(1, 5, 28, 28, device=device)

    layer1.learn(x1, 3)

    x1_1 = layer1 << x1
    print(x1_1.shape)

    layer2.learn(x1_1, 3)

    x1_2 = layer2 << x1_1
    print(x1_2.shape)
    x_ = layer1 >> (layer2 >> x1_2)

    loss = criterion(x_, x1)
    print(loss.item())

    layer1.learn(x2, 3)

    x2_1 = layer1 << x2
    print(x2_1.shape)

    layer2.learn(x2_1, 3)

    hidden = layer2 << (layer1 << x1)
    print(hidden.shape)
    x_ = layer1 >> (layer2 >> hidden)

    loss = criterion(x_, x1)
    print(loss.item())
