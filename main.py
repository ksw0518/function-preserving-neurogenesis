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

# enable venv
# python -m venv venv
# .\venv\Scripts\Activate.ps1

# ===== 실험 설정 (통제 변수는 전부 여기서만 바꿈) =====
CONFIG = {
    'methods': ['ours'],#원래 'standard', 'cbp', 'rejuvenation', 'ours', 'ours_no_noise'
    'seeds': [0], #원래 0,1,2
    'num_tasks': 2,          # Permuted MNIST 과제 수 (계획서: 수백~수천, 일단 작게 시작)
    'epochs_per_task': 1,
    'lr': 0.001,
    'p': 0.05,                # 하위 p% 제거 (1, 5, 10%)
    'noise_std': 0.01,        # 대칭 깨기 노이즈 표준편차
    'beta': 0.99,             # utility EMA 계수
    'maturity': 100,          # 이 step 이상 산 뉴런만 제거 후보
    'k': 50,                  # 재생성 전후 비교 구간 (step)
}

# 랜덤 시드 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# permuted mnist 용 픽셀 섞기
def make_permutation(num_pixels=784):
    """과제 하나에 쓸 고정된 픽셀 순서"""
    return torch.randperm(num_pixels)


def evaluate_all_tasks(model, test_loader, task_permutations, device, criterion):
    accuracies = []

    for task_idx, perm in enumerate(task_permutations):
        acc = tn.evaluate(
            model,
            test_loader,
            criterion=criterion,
            perm=perm

        )

        accuracies.append(acc)

        print(
            f"  Task {task_idx + 1}: {acc:.2f}%"
        )

    return accuracies
# 
def attach_activation_hooks(model, store):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            def make_hook(key):
                def hook(_m, _i, out):
                    store[key] = out.detach()
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles

# 각 뉴런마다 utility랑 age 트랙하기
def initialize_tracking(model):
    names = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    device = next(model.parameters()).device

    #지표 초기화
    #age랑 utility는 layer 이름을 key로 하는 딕셔너리임.

    utility = {name: torch.zeros(linear[i].out_features, device=device)
               for i, name in enumerate(names)}
    age = {name: torch.zeros(linear[i].out_features, dtype=torch.long, device=device)
           for i, name in enumerate(names)}
    return utility, age


def select_low_utility(
    utility, age, maturity, p,
    dead_idx=None,
    use_dead=False
):
    targets = {}
    sources = {}

    for layer_idx, (name, scores) in enumerate(utility.items()):
        mature = age[name] >= maturity

        # dead 뉴런으로 검출
        if use_dead:
            # 기준 age 이상 뉴런들 중 dead 뉴런 선택
            dead = torch.zeros_like(mature, dtype=torch.bool)
            dead[dead_idx[layer_idx]] = True
            candidates = torch.where(mature & dead)[0].tolist()
        # utility 하위 후보로 검출
        # TODO: 하위 후보로 검출하는 로직 구현
        #else:
            #mature = age[layer_idx] >= maturity
            #candidates = torch.where(mature)[0].tolist()

            #candidates = sorted(
            #    candidates,
            #    key=lambda i: scores[i]
            #)

            #num_targets = max(1, int(len(candidates) * p))
            #candidates = candidates[:num_targets]

        targets[name] = candidates

        # 소스 설정: utility가 높은 뉴런
        source_candidates = [
            i for i in range(len(scores))
            if i not in candidates
        ]

        source_candidates.sort(
            key=lambda i: scores[i],
            reverse=True
        )

        # target과 source의 개수를 동일하게 맞춤
        sources[name] = source_candidates[:len(candidates)]
    return targets, sources


def diagnose_and_regrow(method, model, utility, age, optimizer, cfg, log, train_loader, perm):
    """
    매 epoch 끝: utility 하위 p% 진단 → 방법별 재생성 (계획서 1, 2)
    재생성한 step을 log['regrow_steps']에 기록 (지표 1에서 전후 k step 자를 때 사용)
    """
    if method == 'standard':
        return model, optimizer
    
    

    
    elif method == 'ours':
        dead_idx = dt.detect_dead_neuron(
        model, train_loader, perm=perm)
        targets, sources = select_low_utility(
            utility,
            age,
            cfg["maturity"],
            cfg["p"],
            dead_idx=dead_idx,
            use_dead=True
        )

        names = list(utility.keys())
        dead_list = [targets[name] for name in names]
        source_list = [sources[name] for name in names]
        
        
        # 뉴런 교체 정보 출력
        for name in targets:
            for target, source in zip(targets[name], sources[name]):
                print(
                    f"[REGROW] Layer: {name} | "
                    f"Dead neuron: {target} | "
                    f"Cloned from: {source} | "
                    f"Target utility: {utility[name][target].item():.6f} | "
                    f"Source utility: {utility[name][source].item():.6f} | "
                    f"Target age: {age[name][target].item()}"
                )
        
        # 모델 뉴런 삭제/분화
        regrown = rg.regrow(
            model, dead_list, source_list,
            cfg["noise_std"]
        )
        optimizer = optim.Adam(
            regrown.parameters(),
            lr=cfg["lr"]
        )

      
        # 새로 생긴 뉴런 지표 초기화
        for name, indices in targets.items():
            if indices:
                age[name][indices] = 0
                utility[name][indices] = 0

        log["regrow_steps"].append(log["global_step"])
        log["regrow_count"].append(
            sum(len(x) for x in targets.values())
        )
        log["regrow_by_layer"].append(targets)

        return regrown, optimizer
    return model;
    
    

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
    """조건 1개, seed 1개 실험. 기록(log) 반환"""
    set_seed(seed)
    device = net.set_device()
    model = net.MyNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()


    utility, age = initialize_tracking(model)
    activation_store = {}
    handles = attach_activation_hooks(model, activation_store)

    log = {
        "step_acc": [],
        "regrow_steps": [], 
        "regrow_count": [],
        "regrow_by_layer": [], 
        "task_acc": [], 
        "dead_ratio": [],
        "dead_ratio_by_layer": [],
        "task_permutations": [],
        "global_step": 0,

    }
    try:
        for task in range(cfg["num_tasks"]):
            perm = make_permutation()
            log["task_permutations"].append(perm.tolist())
            for _ in range(cfg["epochs_per_task"]):
                def on_step():
                    # 매 stpe마다 utility 업데이트, age 누적
                    tn.update_utility(activation_store, utility, cfg["beta"])
                    for name in age:
                        age[name].add_(1)
                    log["global_step"] += 1

                # 일반 역전파
                tn.train(model, train_loader, criterion, optimizer,
                        perm=perm, log=log, on_step=on_step)
                _, stats = dt.detect_dead_neuron(
                    model, train_loader, perm=perm, return_stats=True
                )
                log["dead_ratio"].append(stats["overall_dead_ratio"])
                log["dead_ratio_by_layer"].append({
                    n: {k: v[k] for k in ("dead_count", "neuron_count", "mean_zero_ratio")}
                    for n, v in stats.items() if n != "overall_dead_ratio"
                })

                old_model = model;
                # 뉴런 삭제, 재생성
                model, optimizer = diagnose_and_regrow(
                method, model, utility, age, optimizer,
                cfg, log, train_loader, perm
                )
                if model is not old_model:
                    # 기존 모델에 등록했던 hook 제거
                    for h in handles:
                        h.remove()

                    # 새 모델에 hook 재등록
                    activation_store.clear()
                    handles = attach_activation_hooks(model, activation_store)

               
            # 정확도 측정
            _, acc = tn.evaluate(model, test_loader, criterion, perm=perm)
            log["task_acc"].append(acc)
            print(f"[{method} | seed {seed}] task {task+1}/{cfg['num_tasks']} acc {acc*100:.2f}%")
    finally:
        for h in handles:
            h.remove()
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