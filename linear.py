import torch
import math
from layer import Layer
import os


class Conceptor(Layer):

    def __init__(self, device, file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.new_weights = []
        self.file_path = file_path

    def save(self):
        if self.file_path:
            torch.save(self.weights, self.file_path)

    def load(self):
        if self.file_path:
            self.weights = torch.load(self.file_path)

    def learn(self, input, expand_depth, expand_threshold=1e-4, steps=1000, lr=0.01, verbose=False):
        print("learn")

        prev_loss = 0
        for k in range(100):

            if len(self.weights) is not 0:
                hidden = self.__internal__forward(input, self.weights)
                input_ = self.__internal__backward(hidden, self.weights, input.shape[1])
            else:
                input_ = torch.zeros(1, input.shape[1], device=self.device)

            with torch.no_grad():
                residue = input - input_

            loss = torch.max(torch.abs(residue))
            if loss.item() < expand_threshold:
                print("Skip expansion after", k, "steps, small reconstruction loss.", loss.item())
                break
            if abs(loss.item() - prev_loss) < expand_threshold:
                print("Skip expansion after", k, "steps, small delta error.", loss.item(), prev_loss)
                break
            prev_loss = loss.item()

            # expand
            A = torch.empty(input.shape[1], expand_depth, device=self.device, requires_grad=True)
            torch.nn.init.xavier_uniform_(A)
            self.new_weights.append(A)

            optimizer = torch.optim.Adam(self.new_weights, lr=lr)
            criterion = torch.nn.MSELoss(reduction='mean')

            for i in range(steps):

                new_hidden = self.__internal__forward(input, self.new_weights)
                residue_ = self.__internal__backward(new_hidden, self.new_weights)

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

            # merge
            self.weights.append(A)
            self.new_weights.clear()

    def __internal__forward(self, input, weights):
        res = torch.cat([
            torch.matmul(input[:, 0:f.shape[0]], f)
            for f in weights
        ], dim=1)
        return res

    def __internal__get_canvas(self, hidden, weights, depth_out=0):

        for f in weights:
            depth_out = max(depth_out, f.shape[0])

        canvas = torch.zeros([hidden.shape[0], depth_out], device=self.device)
        return canvas

    # https://github.com/vdumoulin/conv_arithmetic
    def __internal__backward(self, hidden, weights, depth_out=0):

        canvas = self.__internal__get_canvas(hidden, weights, depth_out)

        from_depth = 0
        for f in weights:
            to_depth = from_depth + f.shape[1]
            addition = torch.matmul(hidden[:, from_depth:to_depth], torch.transpose(f, 0, 1))
            occupied_depth = f.shape[0]
            canvas[:, 0:occupied_depth] = canvas[:, 0:occupied_depth] + addition
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
    print("assert conceptor preserves the containment property")

    dtype = torch.float
    device = torch.device("cuda:0")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    layer1 = Conceptor(device, file_path=os.path.join(dir_path, "weights", "linear_layer1.wt"))
    layer2 = Conceptor(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    x1 = torch.randn(30, 50, device=device)
    x2 = torch.randn(30, 50, device=device)

    layer1.learn(x1, 1)

    x1_1 = layer1 << x1
    print(x1_1.shape)

    layer2.learn(x1_1, 1)

    x1_2 = layer2 << x1_1
    print(x1_2.shape)
    x_ = layer1 >> (layer2 >> x1_2)

    loss = criterion(x_, x1)
    print(loss.item())

    layer1.learn(x2, 1)

    x2_1 = layer1 << x2
    print(x2_1.shape)

    layer2.learn(x2_1, 1)

    hidden = (layer2 << (layer1 << x1))
    print(hidden.shape)
    x_ = layer1 >> (layer2 >> hidden)

    loss = criterion(x_, x1)
    print(loss.item())
