import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from conceptor import Cross_Correlational_Conceptor
from linear import Conceptor
from nearest import Nearest_Neighbor
from transfer import Mirroring_Relu_Layer
from semantic import Semantic_Memory
from dataset import FashionMNIST


class Block_LML:
    def __init__(self):
        self.c0 = Conceptor(device)
        self.t0 = Mirroring_Relu_Layer(device)
        self.c1 = Conceptor(device)

    def __le__(self, input):
        input = torch.reshape(input, [input.shape[0], -1])
        self.c0.learn(input, 1)
        input = self.c0 << input
        input = self.t0 << input

        self.c1.learn(input, 1)
        output = self.c1 << input
        return output

    def __lshift__(self, input):
        input = torch.reshape(input, [input.shape[0], -1])
        input = self.c0 << input
        input = self.t0 << input
        output = self.c1 << input
        return output


class Block_CMC:
    def __init__(self):
        self.c0 = Cross_Correlational_Conceptor(device, kernel_size=(3, 3))
        self.t0 = Mirroring_Relu_Layer(device)
        self.c1 = Cross_Correlational_Conceptor(device, kernel_size=(1, 1))

    def __le__(self, input):
        self.c0.learn(input, 1)
        input = self.c0 << input
        input = self.t0 << input

        self.c1.learn(input, 1)
        output = self.c1 << input
        return output

    def __lshift__(self, input):
        input = self.c0 << input
        input = self.t0 << input
        output = self.c1 << input
        return output


if __name__ == "__main__":
    print("main")

    dtype = torch.float
    device = torch.device("cuda:0")

    batch_size = 1
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=20, seed=10, group_size=1)

    cluster_layers = []

    for i in range(2):
        cluster_layers.append(Block_CMC())

    # final_layer = Semantic_Memory(device)
    final_layer = Nearest_Neighbor(device)

    def forward(input):
        for cluster in cluster_layers:
            input = cluster << input
        input = torch.reshape(input, [input.shape[0], -1])
        prediction = final_layer << input
        return prediction

    percent_correct = 0.0
    for i, (data, label) in enumerate(dataset):
        print("data: ", i)

        # img = np.squeeze(data.numpy())
        # cv2.imshow("sample", img)
        # cv2.waitKey(10)
        input = data.to(device)
        output = label.to(device)

        # online test
        if i > 0:
            prediction = forward(input)
            count_correct = np.sum(prediction.cpu().numpy() == label.numpy())
            percent_correct = 0.99 * percent_correct + 0.01 * count_correct * 100 / batch_size
            print("True: ", label, "Guess: ", prediction, "Percent correct: ", percent_correct)

        # then, learn
        for cluster in cluster_layers:
            input = cluster <= input
        input = torch.reshape(input, [input.shape[0], -1])
        final_layer.learn(input, output, 10)
        prediction = final_layer << input

    count = 0
    for i, (data, label) in enumerate(dataset):
        input = data.to(device)
        output = label.to(device)

        # test
        prediction = forward(input)
        count = count + np.sum(prediction.cpu().numpy() == label.numpy())

    print("Percent correct: ", count * 100 / (len(dataset) * batch_size))
