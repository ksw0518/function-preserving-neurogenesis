import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

print('MPS Available:', torch.backends.mps.is_available())
print('MPS Built:', torch.backends.mps.is_built())

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Device: ', device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./practiceFolder/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./practiceFolder/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

images, labels = next(iter(train_loader))

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)
print('First 10 labels:', labels[0].item())

class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 600)
        self.relu1 = nn.Sigmoid()

        self.fc2 = nn.Linear(600, 400)
        self.relu2 = nn.Sigmoid()

        self.fc3 = nn.Linear(400, 200)
        self.relu3 = nn.Sigmoid()
        self.fc4 = nn.Linear(200, 10)

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

model = TestNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

def train(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

epochs = input('Enter the number of epochs: ')

for epoch in range(int(epochs)):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

plt.plot(range(1, int(epochs)+1), train_loss, color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()