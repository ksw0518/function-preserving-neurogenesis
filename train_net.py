import torch
import torch.nn as nn


def permute(images, perm):
    """
    [B, 1, 28, 28] 이미지의 픽셀 순서를 perm(길이 784 인덱스 텐서)대로 섞음
    Permuted MNIST의 '과제'는 이 perm 하나로 정의됨
    """
    return images.flatten(1).index_select(1, perm).reshape_as(images)

# 각 뉴런이 네트워크에서 얼마나 큰 역할을 하는지 예측
def update_utility(activations, utility, beta, model):
    """EMA: mean(|activation|) * mean(|outgoing weight|), per hidden neuron."""
    
    # 모든 linear, relu 레이어 저장
    # linear = fc1, fc2 etc, relu = relu1, relu2 etc
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    relu_names = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    for i, name in enumerate(relu_names):
        if name not in activations:
            continue
        score = activations[name].abs().mean(0) * linear[i + 1].weight.detach().abs().mean(0)
        if name not in utility:
            utility[name] = score.detach().clone()
        else:
            # 지수 이동 평균
            utility[name].mul_(beta).add_(score.detach(), alpha=1-beta)

def update_utility(activations, utility, beta):
    """
    매 순간기여도를 누적하여 계산(지수이동평균 사용)
    Permuted MNIST 사용시 필요
    activations: 훅에서 저장, utility: 누적 점수, beta: EMA 계수
    """
    for name, acti in activations.items():
        batch_score = acti.mean(dim = 0)

        if name not in utility:
            utility[name] = torch.zeros_like(batch_score)

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

        # TODO: perm이 None이 아니면 permute 적용 (.to(device) 전에 하는 걸 추천)
        if perm is not None:
            images = permute(images, perm)

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

        if log is not None:
            log['step_acc'].append(batch_correct / labels.size(0))

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

            # TODO: perm이 None이 아니면 permute 적용
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

    return avg_loss, accuracy
