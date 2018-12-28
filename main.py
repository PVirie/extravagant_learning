import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from conceptor import Cross_Correlational_Conceptor

root = os.path.dirname(os.path.abspath(__file__))

dtype = torch.float
device = torch.device("cuda:0")

data_set = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)


if __name__ == "__main__":
    print("main")

    layer = Cross_Correlational_Conceptor(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    input_0 = data_set[0][0].unsqueeze(1).to(device)

    for i, (data, label) in enumerate(data_loader):

        input = data.to(device)
        layer.learn(input, 5)

        gen_0 = layer >> (layer << input_0)
        loss_0 = criterion(gen_0, input_0)
        print("The first input loss: ", loss_0.item())

        if i == 100:
            break
