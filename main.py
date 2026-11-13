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
    'epochs_per_task': 2,
    'lr': 0.001,
    'p': 0.05,                # 하위 p% 제거 (1, 5, 10%)
    'noise_std': 0.01,        # 대칭 깨기 노이즈 표준편차
    'beta': 0.99,             # utility EMA 계수
    'maturity': 100,          # 이 step 이상 산 뉴런만 제거 후보
    'k': 50,                  # 재생성 전후 비교 구간 (step)
    'near_dead_threshold' : 0.95
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

# 모델의 activation을 추적할 수 있도록 hook설정
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

# 각 뉴런마다 utility랑 age 트래킹
# age는 뉴런 초기화 이후 지난 step수. 
# utility는 forward를 여러번 진행하며 지수 이동 평균으로 구한 평균 activation값
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


# utility = 각 뉴런별 utility 리스트
# age = 각 뉴런별 age 리스트
# maturity_min = 뉴런 삭제 최소 age값
# utility_min_p = utility 하위 방식으로 검출 시 최소값
# dead_idx = 계산된 dead index 값. 전달 안할시 함수 내에서 계산
# use_dead = true일 시 dead 뉴런 검출, false일시 utility가 작은 뉴런 선택
# source_pool_size = 상위 몇 % 중에서 소스를 선택할지

def select_low_utility(
    utility, age, maturity_min, utility_min_p,
    dead_idx=None,
    use_dead=False,
    source_pool_size = 0.2
):
    targets = {}
    sources = {}

    for layer_idx, (name, scores) in enumerate(utility.items()):
        
        # 일정 이상 age인 뉴런들만 검출
        # 재생성 후 일정 기간동안은 뉴런을 보호하여 뉴런이 학습할 수 있도록 함
        mature = age[name] >= maturity_min
        
        mature_indices = torch.where(mature)[0]
        # dead 뉴런으로 검출
        if use_dead:
            # 기준 age 이상 뉴런들 중 dead 뉴런 선택
            dead = torch.zeros_like(mature, dtype=torch.bool)
            dead[dead_idx[layer_idx]] = True
            candidates = torch.where(mature & dead)[0].tolist()
        # utility 하위 후보로 검출
        else:
            num_targets = max(1, int(len(mature_indices) * utility_min_p))
            k = min(num_targets, len(mature_indices))
            
            # mature 뉴런들의 score 추출
            mature_scores = scores[mature_indices]
            
            # torch 내장 함수 사용하여 상위 k개 뉴런 추출
            # (전체 정렬 후 뽑는 방식은 비효율적)
            _, topk_idx = torch.topk(mature_scores, k=k, largest=False)
            
            # 최종 결과만 리스트로 변환
            candidates = mature_indices[topk_idx].tolist()
            

        targets[name] = candidates

        # 소스 설정: utility가 높은 뉴런
        source_candidates = [
            i for i in range(len(scores))
            if i not in candidates
        ]
        # score 기준으로 내림차순 정리
        source_candidates.sort(
            key=lambda i: scores[i],
            reverse=True
        )

        # target과 source의 개수를 동일하게 맞춤
        top_p = 0.3 
        
        # 상위 p퍼센트중 target 개수만큼 선택
        random.seed(26187)
        top_pool_size = max(len(candidates), int(len(source_candidates) * source_pool_size))

        # 상위 n%에 해당하는 뉴런들만 추려냄
        top_pool = source_candidates[:top_pool_size]

        # 4. target 개수와 동일하게 맞추되, 상위 풀에서 랜덤으로 선택
        target_count = len(candidates)
        sources[name] = random.sample(top_pool, min(target_count, len(top_pool)))
    return targets, sources


def diagnose_and_regrow(method, model, utility, age, optimizer, cfg, log, train_loader, perm, dead_idx_calculated = None):
    """
    매 epoch 끝: utility 하위 p% 진단 → 방법별 재생성 (계획서 1, 2)
    재생성한 step을 log['regrow_steps']에 기록 (지표 1에서 전후 k step 자를 때 사용)
    """
    if method == 'standard':
        return model, optimizer
    
    

    
    elif method == 'ours':
        
        
        # dead idx가 전달되었다면 그대로 사용
        # 전달되지 않았다면 새로 계산
        dead_idx = dt.detect_dead_neuron(
        model, train_loader, perm=perm) if dead_idx_calculated == None else dead_idx_calculated
        
        
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



# 방법, 시드, 설정(config), 학습 로더, 테스트 로더
def run(method, seed, cfg, train_loader, test_loader):
    
    # 사전 설정 (시드, 모델 사전설정)
    set_seed(seed)
    device = net.set_device()
    model = net.MainNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    # utility, age 초기화
    utility, age = initialize_tracking(model)

    # 모델의 activation 추적
    activation_store = {}
    handles = attach_activation_hooks(model, activation_store)


    # 저장할 값들
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
        # 설정 상의 num_task 설정값만큼 반복
        for task in range(cfg["num_tasks"]):
            # 새로운 permutation (태스크) 생성
            perm = make_permutation()

            # permutation 저장하여 실험 재현 가능하게
            log["task_permutations"].append(perm.tolist())
            dead_idx = []
            # 설정된 태스크별 epoch 수만큼 에포크 진행
            for _ in range(cfg["epochs_per_task"]):

                def on_step():
                    # 매 step마다 utility, age 업데이트.
                    tn.update_utility(activation_store, utility, cfg["beta"])
                    for name in age:
                        age[name].add_(1)
                    log["global_step"] += 1
 
                # 일반 역전파
                tn.train(model, train_loader, criterion, optimizer,
                        perm=perm, log=log, on_step=on_step)
                
                
                # dead 뉴런 검출
                dead_idx, stats = dt.detect_dead_neuron(
                    model, train_loader, near_dead_threshold= cfg['near_dead_threshold'], use_near_dead= True, perm=perm, return_stats=True
                )
                
                # dead ratio 를 로그에 저장
                log["dead_ratio"].append(stats["overall_dead_ratio"])
                # 레이어별 dead ration 저장
                log["dead_ratio_by_layer"].append({
                    n: {k: v[k] for k in ("dead_count", "neuron_count", "mean_zero_ratio")}
                    for n, v in stats.items() if n != "overall_dead_ratio"
                })

                old_model = model
                # 뉴런 삭제, 재생성
                model, optimizer = diagnose_and_regrow(
                method, model, utility, age, optimizer,
                cfg, log, train_loader, perm, dead_idx
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
        # handle 삭제
        for h in handles:
            h.remove()
    return log




def main():

    # MNIST 데이터 로더 블러오기
    _, _, train_loader, test_loader = net.load_data()


    # result 에 각 method, seed별 로그 저장.
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