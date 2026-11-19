"""현재 MainNetwork를 위한 Neural Rejuvenation의 MLP 적용 버전.

원리: 비활성 뉴런 제거 -> 살아남은 층 너비에 공통 확대율 적용 -> 재초기화.
자원 한도는 첫 모델의 Linear 가중치와 편향 개수로 정하며 반복 회춘에도 유지한다.
입력 차원과 출력 클래스 수는 유지하고 은닉층 너비만 바뀐다.

논문 전체의 재현은 아니다. BatchNorm gamma/L1 기반 검출, gamma rescaling,
cross-attention은 포함하지 않는다. 기존 detect_dead_neuron의 인덱스를 받는다.
새 뉴런의 입력은 nn.Linear의 기본 분포로 초기화하고, 새 뉴런에서 생존
뉴런으로 나가는 연결은 0으로 시작한다. 생존 뉴런 사이 연결은 그대로 옮긴다.
따라서 제거 대상이 실제로 0일 때 생존 경로의 출력을 보존한다.
near-dead 뉴런을 제거하면 그 뉴런이 켜지는 입력에서는 출력이 바뀔 수 있다.

main.py의 rejuvenation 분기에서 run_rejuvenation() 하나로 전체 과정을 호출한다.

낮은 수준의 함수 사용 예:
    model, optimizer, info = rejuvenate(model, dead_idx, optimizer)
    utility, age = remap_tracking(utility, age, info)
    # 기존 hook 핸들 제거 후 새 model에 hook을 등록해야 한다.

참고: Qiao et al., Neural Rejuvenation, CVPR 2019, section 3.2.
https://arxiv.org/abs/1812.00481
"""

import copy
import inspect
import math
from numbers import Integral

import torch
from torch import nn

import detect_dead as dt
import train_net as tn


def _layers(model):
    """현재 프로젝트의 fc1 -> ReLU -> fc2 -> ReLU -> fc3 구조만 지원."""
    expected = {'fc1', 'fc2', 'fc3', 'relu1', 'relu2'}
    if set(dict(model.named_children())) != expected:
        raise ValueError('fc1, fc2, fc3, relu1, relu2로 된 MainNetwork가 필요합니다.')
    layers = [model.fc1, model.fc2, model.fc3]
    if not all(isinstance(layer, nn.Linear) for layer in layers):
        raise ValueError('fc1, fc2, fc3은 nn.Linear여야 합니다.')
    if not all(isinstance(getattr(model, name), nn.ReLU) for name in ('relu1', 'relu2')):
        raise ValueError('은닉 활성화는 nn.ReLU여야 합니다.')
    if any(layer.bias is None for layer in layers):
        raise ValueError('현재 구현은 편향이 있는 Linear를 지원합니다.')
    if layers[0].out_features != layers[1].in_features or layers[1].out_features != layers[2].in_features:
        raise ValueError('Linear 층의 입력/출력 크기가 연결되지 않습니다.')
    parameters = list(model.parameters())
    if len(parameters) != 6 or any(not p.requires_grad for p in parameters):
        raise ValueError('학습 가능한 Linear 가중치와 편향 6개만 지원합니다.')
    if any(p.device != parameters[0].device or p.dtype != parameters[0].dtype for p in parameters):
        raise ValueError('모든 파라미터의 장치와 dtype이 같아야 합니다.')
    return layers


def _parameter_count(input_size, widths, output_size):
    sizes = [input_size, *widths, output_size]
    return sum((left + 1) * right for left, right in zip(sizes, sizes[1:]))


def _allocate_widths(input_size, surviving, output_size, budget):
    """공통 확대율로 정수 너비를 찾는다. 반올림으로 예산을 넘지 않게 내림한다."""
    def widths_at(scale):
        return [max(count, math.floor(count * scale)) for count in surviving]

    def cost(scale):
        return _parameter_count(input_size, widths_at(scale), output_size)

    if cost(1.0) > budget:
        raise ValueError('생존 뉴런을 유지하기에 파라미터 예산이 부족합니다.')
    low, high = 1.0, 2.0
    while cost(high) <= budget:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle
    return widths_at(low)


def _copy_block(destination, source, output_indices, input_indices):
    """옛 생존 행/열을 새 텐서의 왼쪽 위에 같은 순서로 옮긴다."""
    selected = source[output_indices][:, input_indices]
    destination[:len(output_indices), :len(input_indices)] = selected


def _migrate_adam(optimizer, old_model, new_model, index_pairs):
    """기존 연결의 Adam 이력을 옮기고 새 연결의 모멘트는 0으로 시작."""
    # AdamW가 내부 defaults에만 추가하는 옵션은 생성자에 넘기지 않는다.
    accepted = inspect.signature(type(optimizer)).parameters
    options = {key: value for key, value in optimizer.defaults.items() if key in accepted}
    new_optimizer = type(optimizer)(new_model.parameters(), **options)
    settings = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != 'params'}
    new_optimizer.param_groups[0].update(settings)
    old_parameters = dict(old_model.named_parameters())
    for name, new_parameter in new_model.named_parameters():
        old_parameter = old_parameters[name]
        old_state = optimizer.state.get(old_parameter)
        if not old_state:
            continue
        layer_index = int(name[2]) - 1  # fc1 / fc2 / fc3
        output_indices, input_indices = index_pairs[layer_index]
        new_state = {}
        for key, value in old_state.items():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                buffer = value.new_zeros(new_parameter.shape)
                if name.endswith('.weight'):
                    _copy_block(buffer, value, output_indices, input_indices)
                else:
                    buffer[:len(output_indices)] = value[output_indices]
                new_state[key] = buffer
            else:
                # Adam의 step은 파라미터 전체의 스칼라이므로 기존 값을 유지한다.
                new_state[key] = copy.deepcopy(value)
        new_optimizer.state[new_parameter] = new_state
    return new_optimizer


def rejuvenate(model, dead_idx, optimizer=None, *, parameter_budget=None):
    """두 은닉층의 비활성 인덱스를 받아 (새 모델, 새 Adam 또는 None, 보고) 반환.

    dead_idx는 [[fc1의 제거 번호들], [fc2의 제거 번호들]] 형식.
    optimizer는 모델 전체를 단일 그룹으로 학습하는 Adam/AdamW를 지원한다.
    각 은닉층에 생존 뉴런이 적어도 하나 필요하며 모두 제거하려 하면 오류를 낸다.
    원본 모델/옵티마이저는 보존한다. 제거할 뉴런이 없으면 원래 객체를 반환한다.
    parameter_budget은 최초 호출에서만 지정하고 이후에는 기본값을 쓰면 유지된다.
    """
    layers = _layers(model)
    if len(dead_idx) != 2:
        raise ValueError('dead_idx에는 두 은닉층의 목록이 필요합니다.')
    old_widths = [layer.out_features for layer in layers[:2]]
    surviving_indices = []
    for width, indices in zip(old_widths, dead_idx):
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < width for i in indices):
            raise ValueError('제거 번호는 해당 층 범위 안의 정수여야 합니다.')
        if len(set(indices)) != len(indices):
            raise ValueError('제거 번호가 중복되었습니다.')
        removed = set(indices)
        survivors = [i for i in range(width) if i not in removed]
        if not survivors:
            raise ValueError('공통 비율 재배분에는 층마다 생존 뉴런이 하나 이상 필요합니다.')
        surviving_indices.append(survivors)

    if optimizer is not None:
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
            raise ValueError('현재 구현은 Adam과 AdamW를 지원합니다.')
        if len(optimizer.param_groups) != 1:
            raise ValueError('단일 파라미터 그룹의 Adam이 필요합니다.')
        if {id(p) for p in optimizer.param_groups[0]['params']} != {id(p) for p in model.parameters()}:
            raise ValueError('optimizer가 현재 모델 전체의 파라미터를 참조해야 합니다.')

    input_size, output_size = layers[0].in_features, layers[-1].out_features
    before_count = _parameter_count(input_size, old_widths, output_size)
    budget = parameter_budget
    if budget is None:
        budget = getattr(model, '_nr_parameter_budget', before_count)
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < before_count:
        raise ValueError('parameter_budget은 현재 모델 크기 이상의 정수여야 합니다.')

    surviving = [len(indices) for indices in surviving_indices]
    changed = any(len(indices) for indices in dead_idx)
    widths = _allocate_widths(input_size, surviving, output_size, budget) if changed else old_widths
    info = {
        'changed': changed,
        'old_widths': old_widths,
        'new_widths': widths,
        'survivor_indices': dict(zip(('relu1', 'relu2'), surviving_indices)),
        'removed_count': sum(len(indices) for indices in dead_idx),
        'new_count': sum(width - count for width, count in zip(widths, surviving)),
        'parameter_budget': int(budget),
        'parameters_before': before_count,
        'parameters_after': _parameter_count(input_size, widths, output_size),
    }
    if not changed:
        return model, optimizer, info

    # deepcopy하지 않고 새 Linear만 교체해도 되지만, 원본의 다른 상태도 보존한다.
    new_model = copy.deepcopy(model)
    for module in new_model.modules():
        # 복제된 훅은 옛 activation_store를 가리키므로 제거하고 호출부에서 재등록한다.
        module._forward_hooks.clear()
    new_model._nr_parameter_budget = int(budget)
    sizes = [input_size, *widths, output_size]
    all_indices = [list(range(input_size)), *surviving_indices, list(range(output_size))]
    index_pairs = []
    with torch.no_grad():
        for i, old_layer in enumerate(layers):
            new_layer = nn.Linear(sizes[i], sizes[i + 1], device=old_layer.weight.device, dtype=old_layer.weight.dtype)
            input_indices, output_indices = all_indices[i], all_indices[i + 1]
            # 새 뉴런 -> 생존 뉴런 연결은 0. 새 행은 Linear 기본 초기값 유지.
            new_layer.weight[:len(output_indices)].zero_()
            _copy_block(new_layer.weight, old_layer.weight, output_indices, input_indices)
            new_layer.bias[:len(output_indices)] = old_layer.bias[output_indices]
            new_layer.train(old_layer.training)
            setattr(new_model, f'fc{i + 1}', new_layer)
            index_pairs.append((output_indices, input_indices))
    new_optimizer = None
    if optimizer is not None:
        new_optimizer = _migrate_adam(optimizer, model, new_model, index_pairs)
    return new_model, new_optimizer, info


def remap_tracking(utility, age, info):
    """기존 utility/age를 새 번호로 옮기고 새 뉴런의 두 지표는 0으로 만든다."""
    new_utility, new_age = {}, {}
    for name, width in zip(('relu1', 'relu2'), info['new_widths']):
        indices = info['survivor_indices'][name]
        old_width = info['old_widths'][int(name[-1]) - 1]
        if utility[name].shape != (old_width,) or age[name].shape != (old_width,):
            raise ValueError('utility와 age의 크기가 회춘 전 모델과 맞지 않습니다.')
        new_utility[name] = utility[name].new_zeros(width)
        new_age[name] = age[name].new_zeros(width)
        new_utility[name][:len(indices)] = utility[name][indices]
        new_age[name][:len(indices)] = age[name][indices]
    return new_utility, new_age


def run_rejuvenation(
    model, utility, age, optimizer, cfg, log, train_loader, perm,
    *, test_loader=None, criterion=None, dead_idx_calculated=None,
):
    """epoch 끝에 호출. 검출부터 기록까지 처리하고 (model, optimizer)를 반환.

    충분히 오래된 비활성 뉴런만 제거하며 층 전체가 후보면 utility가 가장 높은
    하나를 남긴다. utility/age 딕셔너리 자체를 유지해서 기존 on_step 콜백과
    연결한다. 새 모델의 훅 등록은 main.py의 모델 변경 처리 블록이 담당한다.
    """
    if dead_idx_calculated is None:
        dead_idx = dt.detect_dead_neuron(
            model, train_loader,
            near_dead_threshold=cfg["near_dead_threshold"],
            use_near_dead=True, perm=perm,
        )
    else:
        dead_idx = dead_idx_calculated

    if len(dead_idx) != 2:
        raise ValueError('두 은닉층의 검출 결과가 필요합니다.')
    targets = {}
    for name, indices in zip(("relu1", "relu2"), dead_idx):
        targets[name] = [
            i for i in indices if age[name][i].item() >= cfg["maturity"]
        ]
        if len(targets[name]) == age[name].numel():
            keep = utility[name].argmax().item()
            targets[name].remove(keep)

    dead_list = [targets["relu1"], targets["relu2"]]
    if not any(dead_list):
        return model, optimizer

    measure_accuracy = criterion is not None and test_loader is not None
    if measure_accuracy:
        _, acc_before = tn.evaluate(model, test_loader, criterion, perm=perm)

    new_model, new_optimizer, info = rejuvenate(model, dead_list, optimizer)
    new_utility, new_age = remap_tracking(utility, age, info)
    utility.clear()
    utility.update(new_utility)
    age.clear()
    age.update(new_age)

    event = {"step": log["global_step"], **info}
    if measure_accuracy:
        _, acc_after = tn.evaluate(new_model, test_loader, criterion, perm=perm)
        delta = (acc_after - acc_before) * 100
        event.update(
            accuracy_before=acc_before, accuracy_after=acc_after,
            delta_percentage_points=delta,
        )
        print(
            f"[REJUVENATION] Step {log['global_step']} | "
            f"{acc_before * 100:.2f}% -> {acc_after * 100:.2f}% "
            f"({delta:+.2f}%p)"
        )

    log["regrow_steps"].append(log["global_step"])
    log["regrow_count"].append(info["removed_count"])
    log["regrow_by_layer"].append(targets)
    log.setdefault("rejuvenation_events", []).append(event)
    return new_model, new_optimizer
