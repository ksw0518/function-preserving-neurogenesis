import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

# MNIST 테스트 데이터
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 데이터를 Batch 단위로 묶기
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

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("First label:", labels[0].item())


class MyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)

        self.relu1 = nn.ReLU()


    def forward(self, x):

        # 이미지 평탄화
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x


model = MyNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

def train(model, dataloader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # 1. Gradient 초기화
        optimizer.zero_grad()

        # 2. Forward
        outputs = model(images)

        # 3. Loss 계산
        loss = criterion(outputs, labels)

        # 4. Backward
        loss.backward()

        # 5. 가중치 업데이트
        optimizer.step()

        # 통계 계산
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy




epochs = 5

for epoch in range(epochs):

    train_loss, train_acc = train(
        model,
        train_loader,
        criterion,
        optimizer
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc*100:.2f}% "
        f"| Test Loss: {test_loss:.4f} "
        f"| Test Acc: {test_acc*100:.2f}%"
    )