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


# dead 뉴런 검출
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

    # training 중이였다면 학습 재개하기 위해서 저장
    was_training = model.training
    
    model.eval()
    
    # 훅 설정 (뉴런의 activation 트래킹하기 위해서)
    hooks = register_hooks(model, target_layers, activations)
    
    # 뉴런이 dataset에서 몇번 0인지 기록
    zero_counts = {}
    
    total_samples = 0

    try:
        # 역전파 안쓸것이므로. (no_grad안하면 역전파 위해 정보를 자동 수집함)
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                
                # dataset 크기만큼 total sample 증가시킴
                total_samples += images.size(0)
                
                # 데이터셋 전체를 순전파 돌림
                model(images)

                # 각 뉴런마다 특정 dataset 단위로 (batch size 개수만큼) 활성이 0이었던 개수 저장
                for name in target_layers:
                    act = activations[name]
                    zero_count = (act == 0).sum(dim=0)
                
                    if name not in zero_counts:
                        zero_counts[name] = torch.zeros_like(zero_count, dtype=torch.long, device="cpu")
                    zero_counts[name] += zero_count.cpu()
    finally:
        
        # 작업 완료 후 hook 제거.
        # hook 이용해서 작업 후에는 필수적으로 제거
        for h in hooks:
            h.remove()
        if was_training:
            model.train()  # 원래 모드로 복구
    # 여기까지 각 뉴런별로 0이었던 횟수 저장.
    # ---------------------
    
    
    
    # 리스트로 저장한 버전
    # [[], []] 형태
    dead_idx = []
    stats = {}

    
    # dead 뉴런 개수
    total_dead = 0
    total_neurons = 0
    
    
    for name in target_layers:  # target_layers 순서대로 → 바깥 인덱스 = 층 번호
        
        # 뉴런마다 0이었던 비율 계산
        zero_ratio = zero_counts[name].float() / total_samples

        # near dead 뉴런 사용 - 일정 비율 이상 죽어있던 비율 로 설정
        detect_threshold = near_dead_threshold if use_near_dead else 1.0
 

        # dead 뉴런 인덱스로 저장
        dead = torch.where(zero_ratio >= detect_threshold)[0]
        
        # 현재 dead 인덱스를 dead_idx에 저장 (즉, 레이어별로 저장됨)
        dead_idx.append(dead.tolist())
        
        # dead 뉴런 개수
        dead_count = len(dead)
        
        # zero_ration는 각 뉴런을 인덱스로 가지는 리스트이므로, 
        # 이 리스트의 길이가 뉴런 개수
        neuron_count = len(zero_ratio)

        # stats 저장. 레이어별로 dead 뉴런 개수, 전체 뉴런 개수, 평균 0 비율 계산
        stats[name] = {
            "dead_count": dead_count,
            "neuron_count": neuron_count,
            "mean_zero_ratio": zero_ratio.mean().item()
        }

        # 전체 네트워크에서 총 dead 뉴런 개수
        total_dead += dead_count
        total_neurons += neuron_count

    
    # 전체 뉴런의 0 비율 평균 (하나의 값)
    stats["overall_dead_ratio"] = (
        total_dead / total_neurons if total_neurons > 0 else 0
    )
    
    if return_stats:
        # stat과 dead idx 함께 반환
        return dead_idx, stats
    
    # dead index 반환
    return dead_idx


if __name__ == '__main__':
    import network as net
    torch.manual_seed(0)  # 테스트 결과가 일정하도록 여기서는 랜덤 시드를 정함.

    device = net.set_device()
    train_dataset, test_dataset, train_loader, test_loader = net.load_data()
    
    
    # 테스트용: 일부러 dead neuron을 만들어놓고 검출 결과와 일치하는지 확인
    model = net.MainNetwork().to(device)
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