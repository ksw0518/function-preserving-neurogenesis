import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# GPU 사용 가능 여부 확인
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 학습 데이터
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# i = 0
# for images, labels in train_loader:
#     print(f'image: {images}, label: {labels}')
#     if i > 10: break
#     i += 1

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 600)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(600, 300)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(300, 100)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)

        return x


model = Network().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)

def train(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    total = 0



