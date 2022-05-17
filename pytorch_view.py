from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



conv1 = nn.Conv2d(3, 80, 2)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(80, 200, 2)
# 200 * 7 * 7 -> es el tamaño resultante de la matriz luego de pasar por todo
fc1 = nn.Linear(9800, 120) 
#
fc2 = nn.Linear(120, 84)
fc3 = nn.Linear(84, 10)
fc4 = nn.Linear(10,5)

def forward(x):
    print("init: ",x.shape)
    x = conv1(x)
    print("conv1: ",x.shape)
    x = F.relu(x)
    x = pool(x)
    print("pool1:", x.shape)
    x = conv2(x)
    print("conv2: ", x.shape)
    x = F.relu(x)
    x = pool(x)
    print("pool2: ", x.shape)
    x = torch.flatten(x, 1)
    print("flatten:", x.shape)
    #x = torch.cat((x,x),1)
    
    x = fc1(x)
    x = F.relu(x)
    x = fc2(x)
    x = F.relu(x)
    x = fc3(x)
    print("fc3: ", x.shape)
    #print(type(x))
    #print(x)
    """x = torch.cat((x,x))
    x = fc4(x)
    print("fc4: ", x.shape)
    print(type(x))"""

#print(trainset)
#print("\n")
print(len(testloader))
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
    forward(inputs)
    if i == 0: break
