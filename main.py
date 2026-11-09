import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import network as net
import train_net as tn
import detect_dead as dt
import regrow as rg


# ===== 실험 설정 (통제 변수는 전부 여기서만 바꿈) =====
CONFIG = {
    'methods': ['standard', 'cbp', 'rejuvenation', 'ours', 'ours_no_noise'],
    'seeds': [0, 1, 2],
    'num_tasks': 10,          # Permuted MNIST 과제 수 (계획서: 수백~수천, 일단 작게 시작)
    'epochs_per_task': 1,
    'lr': 0.001,
    'p': 0.05,                # 하위 p% 제거 (1, 5, 10%)
    'noise_std': 0.01,        # 대칭 깨기 노이즈 표준편차
    'beta': 0.99,             # utility EMA 계수
    'maturity': 100,          # 이 step 이상 산 뉴런만 제거 후보
    'k': 50,                  # 재생성 전후 비교 구간 (step)
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_permutation(num_pixels=784):
    """과제 하나에 쓸 고정된 픽셀 순서"""
    return torch.randperm(num_pixels)


def diagnose_and_regrow(method, model, optimizer, cfg, log):
    """
    매 epoch 끝: utility 하위 p% 진단 → 방법별 재생성 (계획서 1, 2)
    재생성한 step을 log['regrow_steps']에 기록 (지표 1에서 전후 k step 자를 때 사용)
    """
    if method == 'standard':
        return model, optimizer

    # TODO: utility 하위 p% 뉴런 → dead_idx, 상위 뉴런 → neurons_to_clone
    #       형식: [[0번 층 뉴런들], [1번 층 뉴런들]] (같은 층끼리 개수 같게)
    # TODO: method별 분기
    #   'cbp'           → 입력 가중치 초기 분포에서 재샘플, 출력 가중치 0
    #   'rejuvenation'  → NR 방식
    #   'ours'          → rg.regrow(..., noise_std=cfg['noise_std'])
    #   'ours_no_noise' → rg.regrow(..., noise_std=0.0)
    # 주의: rg.regrow는 deepcopy한 새 모델을 반환 → optimizer가 옛 모델을 가리킴
    #       in-place 버전을 쓰거나, optimizer 처리 방식을 5개 조건 모두 동일하게 맞출 것
    # TODO: 재생성한 현재 step 수를 log['regrow_steps']에 append
    return model, optimizer


def run(method, seed, cfg, train_loader, test_loader):
    """조건 1개 × seed 1개 실험. 기록(log) 반환"""
    set_seed(seed)
    device = net.set_device()
    model = net.MyNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    log = {
        'step_acc': [],       # 지표 1: 매 step 학습 정확도
        'regrow_steps': [],   # 지표 1: 재생성 시점
        'task_acc': [],       # 지표 2: 과제별 최종 테스트 정확도
        'dead_ratio': [],     # 지표 3: 매 epoch 비활성 뉴런 비율
    }

    for task in range(cfg['num_tasks']):
        perm = make_permutation()

        for epoch in range(cfg['epochs_per_task']):
            tn.train(model, train_loader, criterion, optimizer, perm=perm, log=log)
            # TODO: utility 갱신 함수 만들면 on_step=... 으로 연결

            # TODO: 지표 3 — dt.detect_dead_neuron으로 비율 계산해서 log['dead_ratio']에 기록
            #       (주의: 지금 detect_dead는 permutation을 모름 → perm 인자 추가 필요)

            model, optimizer = diagnose_and_regrow(method, model, optimizer, cfg, log)

        _, acc = tn.evaluate(model, test_loader, criterion, perm=perm)
        log['task_acc'].append(acc)
        print(f'[{method} | seed {seed}] task {task + 1}/{cfg["num_tasks"]} acc {acc * 100:.2f}%')

    return log


def main():
    _, _, train_loader, test_loader = net.load_data()

    results = {}
    for method in CONFIG['methods']:
        for seed in CONFIG['seeds']:
            results[f'{method}_seed{seed}'] = run(
                method, seed, CONFIG, train_loader, test_loader
            )

    # 그래프·표는 이 파일 따로 불러서 그림 (학습과 시각화 분리)
    with open('results.json', 'w') as f:
        json.dump({'config': CONFIG, 'results': results}, f)


if __name__ == '__main__':
    main()