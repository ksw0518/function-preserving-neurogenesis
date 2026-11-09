import torch


def permute(images, perm):
    """
    [B, 1, 28, 28] 이미지의 픽셀 순서를 perm(길이 784 인덱스 텐서)대로 섞음
    Permuted MNIST의 '과제'는 이 perm 하나로 정의됨
    """
    # TODO: 이미지를 [B, 784]로 펴고 → perm 순서로 열 재배열 → 원래 모양으로 복구
    pass


def train(model, dataloader, criterion, optimizer, perm=None, log=None, on_step=None):
    """
    1 epoch 학습
    perm:    주면 Permuted MNIST (픽셀 섞기), None이면 일반 MNIST
    log:     주면 매 step 정확도를 log['step_acc']에 기록 (평가 지표 1: 재생성 전후 k step 비교용)
    on_step: 주면 매 step 끝에 호출 (계획서 1-나: utility EMA 갱신 자리)
    """
    device = next(model.parameters()).device
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:

        # TODO: perm이 None이 아니면 permute 적용 (.to(device) 전에 하는 걸 추천)

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

        batch_correct = (predictions == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)

        # TODO: log가 None이 아니면 이번 배치 정확도(batch_correct / 배치 크기)를 log['step_acc']에 append

        # TODO: on_step이 None이 아니면 호출
        #       utility 계산엔 이번 step의 활성값이 필요 → forward hook으로 잡은 값을 쓰거나,
        #       on_step에 model을 넘겨서 거기서 처리. 어떤 인자를 넘길지는 utility 함수 설계에 맞출 것

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, perm=None):
    """
    perm: 주면 현재 과제(같은 perm)로 평가. 학습 때와 반드시 같은 perm을 넘길 것
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            # TODO: perm이 None이 아니면 permute 적용

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