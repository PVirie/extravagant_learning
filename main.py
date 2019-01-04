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

data_set = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)


if __name__ == "__main__":
    print("main")

    cluster_layers = []

    for i in range(2):
        layers = []
        layers.append(Cross_Correlational_Conceptor(device, kernel_size=(3, 3), stride=(1, 1)))
        layers.append(Cross_Correlational_Conceptor(device, kernel_size=(5, 5), stride=(2, 2)))
        layers.append(Mirroring_Relu_Layer(device))
        layers.append(Cross_Correlational_Conceptor(device, kernel_size=(1, 1), stride=(1, 1)))
        cluster_layers.append(layers)

    final_layer = Nearest_Neighbor(device)

    def forward(input):
        for cluster in cluster_layers:
            for layer in cluster:
                input = layer << input
        input = torch.reshape(input, [input.shape[0], -1])
        prediction = final_layer << input
        return prediction

    count = 0
    for i, (data, label) in enumerate(data_loader):
        print("data: ", i)
        input = data.to(device)
        output = label.to(device)

        # test
        if i > 0:
            prediction = forward(input)
            if prediction.item() == label.item():
                count = count + 1
            print("True: ", label.item(), "Guess: ", prediction.item(), "Percent correct: ", count * 100 / i)

        # then, learn
        for cluster in cluster_layers:
            cluster[0].learn(input, 16)
            input = cluster[0] << input

            cluster[1].learn(input, 16, lr=0.001, steps=2000, verbose=True)
            input = cluster[1] << input

            input = cluster[2] << input

            cluster[3].learn(input, 8)
            input = cluster[3] << input

        input = torch.reshape(input, [input.shape[0], -1])
        final_layer.learn(input, output, 10)

        if i == 1000:
            break
