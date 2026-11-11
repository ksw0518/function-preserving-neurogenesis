import torch
import torch.nn as nn


def hook_fn(name, store):
    def fn(module, input, output):
        store[name] = output.detach()
    return fn

def register_hooks(model, target_layers, store):
    hooks = []
    for name in target_layers:
        layer = getattr(model, name)  # 이름으로 층 꺼내기 (model.relu1과 같음)
        hooks.append(layer.register_forward_hook(hook_fn(name, store)))
    return hooks

def detect_dead_neuron(model, dataloader, near_dead_threshold=0.9, use_near_dead=False, perm=None, return_stats=False):
    """
    반환: 층별 dead 뉴런 인덱스 리스트 (regrow의 dead_idx 형식)
        예) [[2, 17, 45], [8]] → 0번 층 뉴런 2,17,45 / 1번 층 뉴런 8
    use_near_dead=True면 zero ratio가 near_dead_threshold 이상인 뉴런까지 포함
    """
    activations = {}

    device = next(model.parameters()).device

    # 모델 안의 ReLU 층 이름을 정의된 순서대로 자동 수집
    target_layers = [name for name, m in model.named_modules() if isinstance(m, nn.ReLU)]

    was_training = model.training
    model.eval()

    hooks = register_hooks(model, target_layers, activations)
    zero_counts = {}
    total_samples = 0

    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                total_samples += images.size(0)
                model(images)

                for name in target_layers:
                    act = activations[name]
                    zero_count = (act == 0).sum(dim=0)
                    if name not in zero_counts:
                        zero_counts[name] = torch.zeros_like(zero_count, dtype=torch.long, device="cpu")
                    zero_counts[name] += zero_count.cpu()
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()  # 원래 모드로 복구

    dead_idx = []
    stats = {}

    total_dead = 0
    total_neurons = 0
    for name in target_layers:  # target_layers 순서대로 → 바깥 인덱스 = 층 번호
        zero_ratio = zero_counts[name].float() / total_samples

        if use_near_dead:
            dead = torch.where(zero_ratio >= near_dead_threshold)[0]
        else:
            dead = torch.where(zero_ratio == 1.0)[0]

        #print(f"[{name}] 뉴런 {len(zero_ratio)}개 중 dead {len(dead)}개, "
              #f"평균 zero ratio {zero_ratio.mean().item() * 100:.2f}%")

        dead_idx.append(dead.tolist())  # 텐서 → 파이썬 리스트
        
        dead_count = len(dead)
        neuron_count = len(zero_ratio)

        stats[name] = {
            "dead_count": dead_count,
            "neuron_count": neuron_count,
            "mean_zero_ratio": zero_ratio.mean().item()
        }

        total_dead += dead_count
        total_neurons += neuron_count

    stats["overall_dead_ratio"] = (
        total_dead / total_neurons if total_neurons > 0 else 0
    )

    if return_stats:
        return dead_idx, stats
    
    return dead_idx

if __name__ == '__main__':
    import network as net
    torch.manual_seed(0)  # 테스트 결과가 일정하도록 여기서는 랜덤 시드를 정함.

    device = net.set_device()
    train_dataset, test_dataset, train_loader, test_loader = net.load_data()
    # 테스트용: 일부러 dead neuron을 만들어놓고 검출 결과와 일치하는지 확인
    model = net.MyNetwork().to(device)
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    # 일부러 죽일 뉴런 (정답)
    answer = [[3, 7, 10], [5]]

    with torch.no_grad():
        for layer_idx, neurons in enumerate(answer):
            layers[layer_idx].weight[neurons] = 0   # 입력 무시
            layers[layer_idx].bias[neurons] = -1    # 항상 음수 → ReLU 지나면 0

    dead_idx = detect_dead_neuron(model, test_loader)

    print("정답:", answer)
    print("검출:", dead_idx)
    print("일치:", dead_idx == answer)