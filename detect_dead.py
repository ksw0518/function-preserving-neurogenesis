import torch
import numpy as np
import network as net

activations = {}

def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output.detach()
    return fn

def register_hooks(model):
    hooks = []
    layers = {
        "fc1": model.fc1,
        "relu1": model.relu1,
    
    }
    for name, layer in layers.items():
        hooks.append(
            layer.register_forward_hook(hook_fn(name))
        )
    return hooks


def detect_dead_neuron(model, dataloader, optimizer, near_dead_threshold=0.9):
    model.eval()
    activations.clear()

    hooks = register_hooks(model)
    zero_counts = {}
    total_samples = 0
    
    # 검사하고자 하는 레이어 목록
    target_layers = ["relu1"]

    try:
        with torch.no_grad():
            # 1. 전체 데이터셋에서 각 뉴런이 0이 된 횟수를 누적
            for images, labels in dataloader:
                images = images.to(net.device)
                batch_size = images.size(0)
                total_samples += batch_size

                outputs = model(images)

                for name in target_layers:
                    if name not in activations:
                        continue
                    act = activations[name]

                    # 각 뉴런마다 activation이 0이 된 개수 계산
                    zero_count = (act == 0).sum(dim=0)

                    # 처음 등장한 레이어라면 새로 딕셔너리 key 생성
                    if name not in zero_counts:
                        zero_counts[name] = torch.zeros_like(
                            zero_count,
                            dtype=torch.long,
                            device = "cpu"
                        )

                    # 누적 합산
                    zero_counts[name] += zero_count.cpu()

    finally:
        # 에러가 나든 정상 종료되든 반드시 후크를 제거 (메모리 누수 방지때문에 해야함)
        for h in hooks:
            h.remove()

    # 최종 비율 계산
    results = {}
    for name, counts in zero_counts.items():
        # 0인 비율 계산
        zero_ratio = counts.float() / total_samples

        # 100% 0이면 dead neuron
        dead = torch.where(zero_ratio == 1.0)[0]

        # 설정한 임곗값 이상 0인 경우 near dead
        near_dead = torch.where(
            zero_ratio >= near_dead_threshold
        )[0]

        print(f"\n===== {name} =====")
        print("뉴런 개수:", len(zero_ratio))
        print("Dead Neurons:", len(dead))
        print(f"Near-dead Neurons ({near_dead_threshold * 100}% 이상):", len(near_dead))
        print("평균 Zero Ratio:", zero_ratio.mean().item() * 100)
        print("Dead neuron indices:", dead.tolist())
        print("Near-dead neuron indices:", near_dead.tolist())

        # 최종 비율을 결과 딕셔너리에 저장
        results[name] = zero_ratio

    return results