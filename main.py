import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import network as net
import train_net as tn
import detect_dead as dt


epochs = 2
device = net.set_device()
train_dataset, test_dataset, train_loader, test_loader = net.load_data()
model = net.MyNetwork().to(device)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):

    train_loss, train_acc = tn.train(
        model,
        train_loader,
        criterion,
        optimizer
    )

    test_loss, test_acc = tn.evaluate(
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

print(dt.detect_dead_neuron(model, test_loader, optimizer, 0.9))
print("lel")
