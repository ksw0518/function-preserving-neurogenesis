import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


def regrow(model, dead_idx: list[list], neurons_to_clone: list[list], noise_std: float = 0.01):
    """
    뉴런 재생성 함수,
    dead_idx와 neurons_to_clone는 2차원 array(예: [[0, 1], [0, 1]]),
    dead_idx와 neurons_to_clone은 shape가 같아야 함.
    """
    check_shape(model, dead_idx, neurons_to_clone)

    # model 안의 Linear 층만 순서대로 모음 → layers[0]이 첫 번째 Linear
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    # 죽은 뉴런의 원래 가중치 기록 (현재 아래 로직에서는 사용하지 않음)
    dead_neurons = []
    for layer_idx, neurons in enumerate(dead_idx):  # dead_idx 길이만큼만 돎 → 뒤쪽 층은 안 건드림
        layer = layers[layer_idx]
        weights = layer.weight[neurons]  # weight의 행 = 뉴런 하나. 리스트 인덱싱이라 복사본이 나옴
        biases = layer.bias[neurons]

        for w, b in zip(weights, biases):
            layer_dead = {'weight': w, 'bias': b}
            dead_neurons.append(layer_dead)

    # 원본 model은 보존하고 복사본만 수정 (함수 인자로 넘긴 객체를 직접 고치면 원본도 바뀜)
    regrown = copy.deepcopy(model)
    # 위 layers는 원본 model의 층이므로, 복사본 기준으로 다시 모음
    layers = [m for m in regrown.modules() if isinstance(m, nn.Linear)]

    # 가중치를 직접 수정할 땐 학습 추적을 꺼야 에러가 안 남
    with torch.no_grad():
        for layer_idx, (dead_list, src_list) in enumerate(zip(dead_idx, neurons_to_clone)):
            layer = layers[layer_idx]
            # 출력 가중치는 다음 층에 있음 → 마지막 층은 대상이 될 수 없음 (check_shape에서 막음)
            next_layer = layers[layer_idx + 1]

            for d, s in zip(dead_list, src_list):  # d: 죽은 뉴런 자리, s: 복제할 소스 뉴런
                # 입력 가중치·편향을 소스와 동일하게 복사
                # weight[d]는 행 = 이전 층에서 뉴런 d로 들어오는 가중치들
                layer.weight[d] = layer.weight[s]
                layer.bias[d] = layer.bias[s]

                # 출력 가중치를 반씩 분배
                # next_layer.weight[:, s]는 열 = 뉴런 s에서 다음 층 각 뉴런으로 나가는 가중치들
                # s와 d가 같은 값을 내므로 반씩 나누면 다음 층 입력 합이 그대로 유지됨
                half = next_layer.weight[:, s] / 2
                next_layer.weight[:, s] = half
                next_layer.weight[:, d] = half

                # 대칭 깨기: s에는 +x, d에는 -x
                # 둘이 완전히 같으면 학습 때 기울기도 같아서 영원히 같은 뉴런으로 남음
                # +x, -x로 주면 ReLU 양수 구간에선 합이 그대로라 기능이 거의 보존됨
                x = torch.randn_like(layer.weight[s]) * noise_std  # 평균 0, 표준편차 noise_std
                layer.weight[s] += x
                layer.weight[d] -= x

    return regrown


def check_shape(model, dead_idx, neurons_to_clone):
    # 층 개수 비교
    if len(dead_idx) != len(neurons_to_clone):
        raise ValueError('dead_idx랑 neurons_to_clone이랑 꼬라지 맞춰 와.')

    # 층별 뉴런 개수 비교 (층끼리는 개수가 달라도 됨)
    for d, c in zip(dead_idx, neurons_to_clone):
        if len(d) != len(c):
            raise ValueError('dead_idx랑 neurons_to_clone이랑 꼬라지 맞춰 와.')

    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    # -1: 마지막(출력) 층은 다음 층이 없어서 재생성 불가
    if len(dead_idx) > len(layers) - 1:
        raise ValueError('model 꼬라지랑 인덱스 인자 꼬라지 맞춰 와.')

    # out_features = 그 층의 뉴런 개수 → 뉴런 번호는 0 ~ out_features-1
    for i, (d, c) in enumerate(zip(dead_idx, neurons_to_clone)):
        if len(d) == 0:
            continue  # dead 뉴런 없는 층은 건너뜀 (빈 리스트에 max 쓰면 에러)
        if max(max(d), max(c)) >= layers[i].out_features:
            raise ValueError('model 꼬라지랑 인덱스 인자 꼬라지 맞춰 와.')


if __name__ == '__main__':
    import network as net
    device = net.set_device()
    train_dataset, test_dataset, train_loader, test_loader = net.load_data()
    model = net.MyNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    # 주의: model의 파라미터에 연결됨. regrown은 deepcopy된 별개 객체라
    # regrown을 학습시키려면 regrown.parameters()로 새로 만들어야 함
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # 행 i = i번째 Linear 층. 같은 위치끼리 짝: 0번 층 뉴런 0 ← 5 복제, 뉴런 1 ← 6 복제
    neuron_indices = [[0, 1], [0]]
    to_clone_indices = [[5, 6], [5]]

    regrown = regrow(model, neuron_indices, to_clone_indices)

    # 테스트 데이터에서 첫 배치 하나만 꺼냄 (images: [배치, 1, 28, 28])
    images, labels = next(iter(test_loader))
    images = images.to(device)

    # 같은 이미지를 두 모델에 넣어 출력(logits, [배치, 10]) 비교
    out1 = model(images)
    out2 = regrown(images)
    # 가장 크게 달라진 점수 차이. 0에 가까울수록 기능 보존이 잘 된 것
    print((out1 - out2).abs().max())