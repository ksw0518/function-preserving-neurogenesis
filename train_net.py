import torch
import network as net

def train(model, dataloader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(net.device)
        labels = labels.to(net.device)

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

            images = images.to(net.device)
            labels = labels.to(net.device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
