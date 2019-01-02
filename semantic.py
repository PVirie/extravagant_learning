import torch
import torchvision
import math


# simply 1-nn network
class Semantic_Memory:

    def __init__(self, device):
        print("init")
        self.device = device
        self.weights = []

    def learn(self, input, output, num_classes, expand_threshold=1e-2, steps=1000, lr=0.01):
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
        new_weight = (torch.transpose(input, 0, 1), output)

        # merge
        self.weights.append(new_weight)

    def __internal__forward(self, input, weights, depth_out=0):

        logits = - torch.cat([
            torch.sum(input * input, dim=1, keepdim=True) - 2 * torch.matmul(input[:, :A.shape[0]], A) + torch.sum(A * A, dim=0, keepdim=True)
            for (A, B) in weights
        ], dim=1)

        return logits

    # ----------- public functions ---------------

    def __lshift__(self, input):
        with torch.no_grad():
            logits_ = self.__internal__forward(input, self.weights)

            indices = torch.argmax(logits_, dim=1)

            bases = torch.cat([
                B for (A, B) in self.weights
            ], dim=0)

            prediction = bases[indices]

        return prediction


if __name__ == '__main__':
    print("test semantic memory")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Semantic_Memory(device)

    x = torch.randn(10, 5, device=device)
    y = torch.randint(5, (10, ), dtype=torch.int64, device=device)

    layer.learn(x, y, num_classes=5)

    y_ = layer << x
    print(y)
    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() / x.shape[0])

    x2 = torch.randn(20, 10, device=device)
    y2 = torch.randint(10, (20, ), dtype=torch.int64, device=device)

    layer.learn(x2, y2, num_classes=10)

    x3 = torch.randn(20, 10, device=device)
    y3 = torch.randint(10, (20, ), dtype=torch.int64, device=device)

    layer.learn(x3, y3, num_classes=10)

    xs = torch.zeros(x.shape[0], x2.shape[1], device=device)
    xs[:, 0:x.shape[1], ...] = x
    y_ = layer << xs
    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() / x.shape[0])
