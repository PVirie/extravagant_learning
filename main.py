import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from conceptor import Cross_Correlational_Conceptor
from nearest import Nearest_Neighbor
from transfer import Mirroring_Relu_Layer
from semantic import Semantic_Memory

root = os.path.dirname(os.path.abspath(__file__))

dtype = torch.float
device = torch.device("cuda:0")

batch_size = 1
data_set = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print("main")

    cluster_layers = []

    for i in range(2):
        layers = []
        layers.append(Cross_Correlational_Conceptor(device, kernel_size=(3, 3)))
        layers.append(Mirroring_Relu_Layer(device))
        layers.append(Cross_Correlational_Conceptor(device, kernel_size=(1, 1)))
        cluster_layers.append(layers)

    final_layer = Nearest_Neighbor(device)

    def forward(input):
        for cluster in cluster_layers:
            for layer in cluster:
                input = layer << input
        input = torch.reshape(input, [input.shape[0], -1])
        prediction = final_layer << input
        return prediction

    memory_test_list = []

    count = 0
    for i, (data, label) in enumerate(data_loader):
        print("data: ", i)
        memory_test_list.append((data, label))
        input = data.to(device)
        output = label.to(device)

        # online test
        if i > 0:
            prediction = forward(input)
            count = count + np.sum(prediction.cpu().numpy() == label.numpy())
            print("True: ", label, "Guess: ", prediction, "Percent correct: ", count * 100 / (i * batch_size))

        # then, learn
        for cluster in cluster_layers:
            cluster[0].learn(input, 3)
            input = cluster[0] << input
            input = cluster[1] << input

            cluster[2].learn(input, 1)
            input = cluster[2] << input

        input = torch.reshape(input, [input.shape[0], -1])
        final_layer.learn(input, output, 10)

        if i == 1000:
            break

    count = 0
    for i, (data, label) in enumerate(memory_test_list):
        input = data.to(device)
        output = label.to(device)

        # test
        prediction = forward(input)
        count = count + np.sum(prediction.cpu().numpy() == label.numpy())

        if i == 1000:
            print("True: ", label, "Guess: ", prediction, "Percent correct: ", count * 100 / ((i + 1) * batch_size))
            break
