import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import network as net
import train_net as tn


epochs = 5

for epoch in range(epochs):

    train_loss, train_acc = tn.train(
        net.model,
        net.train_loader,
        net.criterion,
        net.optimizer
    )

    test_loss, test_acc = tn.evaluate(
        net.model,
        net.test_loader,
        net.criterion
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc*100:.2f}% "
        f"| Test Loss: {test_loss:.4f} "
        f"| Test Acc: {test_acc*100:.2f}%"
    )