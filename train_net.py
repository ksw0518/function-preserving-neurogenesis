import torch
import torch.nn as nn


def permute(images, perm):
    """
    [B, 1, 28, 28] 이미지의 픽셀 순서를 perm(길이 784 인덱스 텐서)대로 섞음
    Permuted MNIST의 '과제'는 이 perm 하나로 정의됨
    """
    return images.flatten(1).index_select(1, perm).reshape_as(images)


def update_utility(activations, utility, beta):
    """
    매 순간기여도를 누적하여 계산(지수이동평균 사용)
    Permuted MNIST 사용시 필요
    activations: 훅에서 저장, utility: 누적 점수, beta: EMA 계수
    """
    for name, acti in activations.items():
        # 특정 뉴런의 activation의 평균을 사용.
        batch_score = acti.mean(dim = 0)

        # 처음 layer에 대해서 계산할 때에 0으로 초기화
        if name not in utility:
            utility[name] = torch.zeros_like(batch_score)

        # 지수 이동 평균. 현재 score와 기존 utility 값을 조화하여 누적시킴
        utility[name] = beta * utility[name] + (1 - beta) * batch_score





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

        # perm에 따라 이미지 조절
        if perm is not None:
            images = permute(images, perm)

        # gpu(또는 cpu)로 전송
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

        # 매 step별 정확도 기록
        if log is not None:
            log['step_acc'].append(batch_correct / labels.size(0))

        # 스탭별로 on_step함수 호출 (age, utility 업데이트)
        if on_step is not None:
            on_step()


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

            # perm이 None이 아니면 permute 적용
            if perm is not None:
                images = permute(images, perm)

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
    # print(accuracy)  # asdfasdfasdf
    return avg_loss, accuracy
