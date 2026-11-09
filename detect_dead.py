import torch
import torch.nn as nn
import network as net

activations = {}

def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output.detach()
    return fn

def register_hooks(model, target_layers):
    hooks = []
    for name in target_layers:
        layer = getattr(model, name)  # 이름으로 층 꺼내기 (model.relu1과 같음)
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    return hooks


def detect_dead_neuron(model, dataloader, near_dead_threshold=0.9, use_near_dead=False):
    """
    반환: 층별 dead 뉴런 인덱스 리스트 (regrow의 dead_idx 형식)
        예) [[2, 17, 45], [8]] → 0번 층 뉴런 2,17,45 / 1번 층 뉴런 8
    use_near_dead=True면 zero ratio가 near_dead_threshold 이상인 뉴런까지 포함
    """
    # 모델 안의 ReLU 층 이름을 정의된 순서대로 자동 수집
    target_layers = [name for name, m in model.named_modules() if isinstance(m, nn.ReLU)]

    was_training = model.training
    model.eval()
    activations.clear()

    hooks = register_hooks(model, target_layers)
    zero_counts = {}
    total_samples = 0

    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(net.device)
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
    for name in target_layers:  # target_layers 순서대로 → 바깥 인덱스 = 층 번호
        zero_ratio = zero_counts[name].float() / total_samples

        if use_near_dead:
            dead = torch.where(zero_ratio >= near_dead_threshold)[0]
        else:
            dead = torch.where(zero_ratio == 1.0)[0]

        print(f"[{name}] 뉴런 {len(zero_ratio)}개 중 dead {len(dead)}개, "
              f"평균 zero ratio {zero_ratio.mean().item() * 100:.2f}%")

        dead_idx.append(dead.tolist())  # 텐서 → 파이썬 리스트

    return dead_idx