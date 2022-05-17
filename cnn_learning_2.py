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

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

#imshow(torchvision.utils.make_grid(images))
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#print(trainset)


class MyFirstCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # first segment
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # second segment
        self.conv3 = nn.Conv2d(3, 6, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(6, 16, 5)

        # third segment
        self.fc1 = nn.Linear(16*5*5*2, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 42)
        self.fc5 = nn.Linear(42, 10)
    
    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = torch.flatten(x1, 1) #flatten all dimensions except batch

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1,x2),1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

cnn = MyFirstCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)


#TRAIN NN
for epoch in range(2):
    running_loss= 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = cnn(inputs, inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')

PATH_MODEL = './cnn_trained.pth'
torch.save(cnn.state_dict(), PATH_MODEL)

"""#TEST NN
dataiter = iter(testloader)
images, labels = dataiter.next()

PATH_MODEL = './cnn_trained.pth'
trainednet = MyFirstCNN()
trainednet.load_state_dict(torch.load(PATH_MODEL))

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = trainednet(images,images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = trainednet(images, images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')"""
