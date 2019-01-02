import torch
import torchvision


class Semantic_Memory:

    def __init__(self, device):
        print("init")
        self.device = device
        self.weights = []
        self.new_weights = []

    # be careful before stacking this layer to other expandable convolutional layers.
    def learn(self, input, output, num_classes, expand_threshold=1e-6, steps=1000, lr=0.01):
        print("learn")

        with torch.no_grad():
            if len(self.weights) is not 0:
                prev_logits_ = self.__internal__forward(input, self.weights, input.shape[1])
            else:
                prev_logits_ = torch.zeros(input.shape[0], num_classes, device=self.device)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(prev_logits_, output)
        if loss < expand_threshold:
            print("Small error, skip expansion.")
            return

        # expand
        A = torch.empty(input.shape[1], num_classes, device=self.device, requires_grad=True)
        torch.nn.init.normal_(A, 0, 0.001)
        self.new_weights.append(A)

        optimizer = torch.optim.Adam(self.new_weights, lr=lr)

        for i in range(steps):

            logits_ = self.__internal__forward(input, self.new_weights)

            loss = criterion(logits_ + prev_logits_, output)

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

        for f in weights:
            if input.shape[1] < f.shape[0]:
                continue
            addition = torch.matmul(input[:, :f.shape[0]], f)
            occupied_depth = f.shape[1]
            canvas[:, 0:occupied_depth] = canvas[:, 0:occupied_depth] + addition

        return canvas

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            logits_ = self.__internal__forward(input, self.weights)
            prediction = torch.argmax(logits_, dim=1)
        return prediction


if __name__ == '__main__':
    print("test semantic memory")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Semantic_Memory(device)

    x = torch.randn(20, 5, device=device)
    y = torch.randint(5, (20, ), dtype=torch.int64, device=device)

    layer.learn(x, y, num_classes=5)

    y_ = layer << x

    print(y)
    print(y_)

    x2 = torch.randn(20, 10, device=device)
    y2 = torch.randint(10, (20, ), dtype=torch.int64, device=device)

    layer.learn(x2, y2, num_classes=10)

    y_ = layer << torch.cat([x, torch.zeros(x.shape, device=device)], dim=1)

    print(y)
    print(y_)
