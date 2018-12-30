import torch
import torchvision


class Semantic_Memory:

    def __init__(self, device):
        print("init")
        self.device = device
        self.weights = []
        self.new_weights = []

    # be careful before stacking this layer to other expandable convolutional layers.
    def learn(self, input, output, steps=1000, lr=0.01):
        print("learn")

        if len(self.weights) is not 0:
            output_ = self.__internal__forward(input, self.weights, input.shape[1])
        else:
            output_ = torch.zeros(1, output.shape[1], device=self.device)

        # expand
        A = torch.empty(input.shape[1], output.shape[1], device=self.device, requires_grad=True)
        torch.nn.init.normal_(A, 0, 0.001)
        self.new_weights.append(A)

        optimizer = torch.optim.Adam(self.new_weights, lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')

        with torch.no_grad():
            residue = output - output_

        for i in range(steps):

            residue_ = self.__internal__forward(input, self.new_weights)

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

    def __internal__forward(self, input, weights, depth_out=0):

        for f in weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([input.shape[0], depth_out], device=self.device)

        from_depth = 0
        for f in weights:
            to_depth = from_depth + f.shape[0]
            addition = torch.matmul(input, f)
            occupied_depth = f.shape[1]
            canvas[:, 0:occupied_depth] = canvas[:, 0:occupied_depth] + addition
            from_depth = to_depth

        return canvas

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            res = self.__internal__forward(input, self.weights)
        return res


if __name__ == '__main__':
    print("test semantic memory")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Semantic_Memory(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    x = torch.randn(1, 10, device=device)
    y = torch.randn(1, 5, device=device)

    layer.learn(x, y)

    y_ = layer << x

    loss = criterion(y_, y)
    print(loss.item())
