from random import shuffle
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time


class SecondCNN(nn.Module):
    def __init__(self, numChannels, classes):
        super(SecondCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output

# extras
# construct the argument parser and parse the arguments
"""ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())"""

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 2#10
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#print(device)

trainData = torchvision.datasets.KMNIST(
    root="./data_2", 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)

testData = torchvision.datasets.KMNIST(
    root="./data_2", 
    train=False, 
    download=True,
    transform=transforms.ToTensor()
)

numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = torch.utils.data.random_split(
    trainData, 
    [numTrainSamples, numValSamples], 
    generator=torch.Generator().manual_seed(42)
)

trainDataLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = torch.utils.data.DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size=BATCH_SIZE)

trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

model = SecondCNN(numChannels=1, classes=len(trainData.dataset.classes)).to(device)
opt = optim.Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

startTime = time.time()

for e in range(0, EPOCHS):
    # training mode
    model.train()

    # init total trianing and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # init correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for (x, y) in trainDataLoader:
        (x, y) = (x.to(device), y.to(device))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)

        # zero out the gradients, perform the backpropagation step,
		# and update the weights
        opt.zero_grad()
        loss.backward()
