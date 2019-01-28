import torch
from layer import *
import os


class Conceptor(Layer):

    def __init__(self, device, file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.file_path = file_path
        self.max_input_channel = 0

    def save(self):
        if self.file_path:
            torch.save(self.weights, self.file_path)

    def load(self):
        if self.file_path:
            self.weights = torch.load(self.file_path)

    def learn(self, input, expand_depth=1, expand_threshold=1e-4, expand_steps=1000, steps=1000, lr=0.01, verbose=False):
        print("learn")

        self.max_input_channel = max(self.max_input_channel, input.shape[1])

        criterion = torch.nn.MSELoss(reduction='mean')

        prev_size = len(self.weights)
        prev_loss = 0
        for k in range(expand_steps):

            with torch.no_grad():
                if len(self.weights) is not 0:
                    hidden = self.__internal__forward(input, self.weights)
                    input_ = self.__internal__backward(hidden, self.weights, input.shape[1])
                else:
                    input_ = torch.zeros(1, input.shape[1], device=self.device)

                residue = input - input_

            rloss = criterion(input_, input)
            if rloss.item() < expand_threshold:
                print("Stop expansion after", (len(self.weights) - prev_size) * expand_depth, "steps, small reconstruction loss.", rloss.item())
                break
            if abs(rloss.item() - prev_loss) < expand_threshold:
                print("Stop expansion after", (len(self.weights) - prev_size) * expand_depth, "steps, small delta error.", rloss.item(), prev_loss)
                break

            # expand
            A = torch.empty(input.shape[1], expand_depth, device=self.device, requires_grad=False)

            with torch.no_grad():
                AA = torch.matmul(torch.transpose(residue, 0, 1), residue)
                U, S, V = torch.svd(AA)
                A_ = V[:, 0:expand_depth]
                A.copy_(A_)

            check = S[expand_depth - 1].item()
            if check * check < expand_threshold:
                print("Failed solution, continue...", check)
                continue

            # merge
            self.weights.append(A)
            prev_loss = rloss.item()

    def __internal__forward(self, input, weights):
        res = torch.cat([
            torch.matmul(input[:, 0:f.shape[0]], f)
            for f in weights
        ], dim=1)
        return res

    def __internal__get_canvas(self, hidden, weights, depth_out=0):

        depth_out = max(depth_out, self.max_input_channel)
        for f in weights:
            depth_out = max(depth_out, f.shape[0])

        canvas = torch.zeros([hidden.shape[0], depth_out], device=self.device)
        return canvas

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

    # if the model is correctly implemented, the total number of steps should not exceed min(x1.shape[0], x1.shape[1])
    x1 = torch.randn(20, 30, device=device)
    x2 = torch.randn(20, 30, device=device)

    layer1.learn(x1, 1)

    x1_1 = layer1 << x1

    layer2.learn(x1_1, 1)

    x1_2 = layer2 << x1_1
    x_ = layer1 >> (layer2 >> x1_2)

    loss = criterion(x_, x1)
    print(loss.item())

    layer1.learn(x2, 1)

    x2_1 = layer1 << x2

    layer2.learn(x2_1, 1)

    hidden = (layer2 << (layer1 << x1))
    x_ = layer1 >> (layer2 >> hidden)

    loss = criterion(x_, x1)
    print(loss.item())
