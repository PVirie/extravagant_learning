import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))

dtype = torch.float
device = torch.device("cuda:0")

data_set = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)


num_layers = 5

inputs = data_set[0][0].unsqueeze(1).to(device)
filters = torch.randn(5, 1, 3, 3, device=device)
res = torch.nn.functional.conv2d(inputs, filters, padding=1)

print(res.shape)

A = torch.randn(5, 3, 3, 3, device=device)
B = torch.randn(5, 5, 3, 3, device=device)

weights = [A, B]

res = torch.cat([torch.nn.functional.conv2d(res[:, 0:f.shape[1], ...], f, padding=1) for f in weights], dim=1)

optimizer = torch.optim.Adam(weights, lr=0.0001)

print(res.shape)

if __name__ == "__main__":
    print("main")

    # for i, (data, label) in enumerate(data_loader):
