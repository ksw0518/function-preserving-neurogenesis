"""Continual ImageNet 전용 실행 파일. MNIST main.py/config.json과 독립적으로 실행.

준비: ImageNet train/<class>/*, val/<class>/* 폴더 또는 data/*.parquet.
Parquet는 첫 실행 시 외장 SSD에 uint8 메모리맵 캐시를 만든다. pyarrow 필요.
이미지는 RGB 32x32로 변환한다. 라벨 없는 이미지 모음은 사용할 수 없다.
실행: python3 -B imagenet_main.py --data-root /path/to/imagenet
검증: python3 -B imagenet_main.py --smoke-test --output-dir /tmp/imagenet-check
그래프만: python3 -B imagenet_main.py --plot-only /path/to/finished-run

과제마다 중복 없는 클래스 쌍을 선택하고 0/1로 라벨을 바꾼다.
모델/Adam은 과제 간 유지하며 모든 방법/시드는 동일한 과제 순서로 비교한다.
ours와 ours_no_noise는 매 epoch 성숙 채널 중 utility 하위 p 비율을 골라 상위 utility 채널로 복제한다.
비활성 비율은 교체 전 측정하며, rejuvenation은 성숙한 비활성 채널에 재배분을 적용한다.
cbp는 매 epoch 성숙 채널의 utility 하위 p 비율을 무작위 재초기화하고 출력을 0으로
설정하는 계획서 기준 변형이다. 원 논문의 매 step 교체 스케줄 전체 재현은 아니다.
회춘은 기존 MLP 버전과 같은 NR 적용 버전이며 BN gamma/L1/rescaling 및
cross-attention을 포함하는 원 논문의 전체 재현이 아니다.
"""

import argparse
import csv
import os
import tempfile
import fcntl
import io
import json
import math
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

import network as net
import train_net as tn


class ImageNetCNN(nn.Module):
    """모든 비교 방법이 같은 초기 구조를 쓰는 3층 CNN."""
    def __init__(self, widths=(16, 32, 64)):
        super().__init__()
        self.widths = tuple(widths)
        self.convs = nn.ModuleList([
            nn.Conv2d(left, right, 3, padding=1)
            for left, right in zip((3, *widths[:-1]), widths)
        ])
        self.relu = nn.ModuleList([nn.ReLU() for _ in widths])
        self.pool = nn.MaxPool2d(2)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], 2)
        self.activations = []

    def forward(self, images):
        self.activations = []
        for i, (conv, relu) in enumerate(zip(self.convs, self.relu)):
            images = relu(conv(images))
            self.activations.append(images.detach())
            if i < len(self.convs) - 1:
                images = self.pool(images)
        return self.classifier(self.average(images).flatten(1))


class ClassPair(Dataset):
    """원래 클래스 번호를 과제 안의 0/1로 바꾼다."""
    def __init__(self, base, pair, class_indices):
        self.base = base
        self.labels = {pair[0]: 0, pair[1]: 1}
        self.indices = class_indices[pair[0]] + class_indices[pair[1]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        return image, self.labels[int(label)]


def index_classes(targets, count):
    indices = [[] for _ in range(count)]
    for index, label in enumerate(targets):
        indices[int(label)].append(index)
    if any(not group for group in indices):
        raise ValueError('학습/평가 데이터에 각 클래스가 한 장 이상 필요합니다.')
    return indices


class CachedImageNet32(Dataset):
    """Parquet를 한번 디코딩해 SSD에 저장하고, 샘플별로 필요한 페이지만 읽는다."""
    def __init__(self, root, split):
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError('Parquet 로딩에는 pyarrow 설치가 필요합니다.') from error
        files = sorted(p for p in (root / 'data').glob(f'{split}-*.parquet')
                       if not p.name.startswith('._'))
        if not files:
            raise ValueError(f'{split} Parquet 파일이 없습니다.')
        cache = root / 'numpy_cache'
        cache.mkdir(exist_ok=True)
        image_path = cache / f'{split}_images.npy'
        label_path = cache / f'{split}_labels.npy'
        marker = cache / f'{split}_complete.json'
        signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        signature = json.loads(json.dumps(signature))
        with (cache / 'conversion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ready = marker.exists() and image_path.exists() and label_path.exists()
            if not ready or json.loads(marker.read_text()).get('source') != signature:
                rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
                image_tmp = cache / f'{split}_images.partial.npy'
                label_tmp = cache / f'{split}_labels.partial.npy'
                images = np.lib.format.open_memmap(image_tmp, mode='w+', dtype=np.uint8,
                                                   shape=(rows, 3, 32, 32))
                labels = np.lib.format.open_memmap(label_tmp, mode='w+', dtype=np.int64, shape=(rows,))
                offset = 0
                for path in files:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['image', 'label']):
                        data = batch.to_pydict()
                        for encoded, label in zip(data['image'], data['label']):
                            with Image.open(io.BytesIO(encoded['bytes'])) as image:
                                image = image.convert('RGB')
                                if image.size != (32, 32):
                                    raise ValueError('32x32가 아닌 이미지가 있습니다.')
                                images[offset] = np.asarray(image).transpose(2, 0, 1)
                            labels[offset] = label
                            offset += 1
                    print(f'[CACHE] {split}: {offset}/{rows}', flush=True)
                images.flush()
                labels.flush()
                del images, labels
                image_tmp.replace(image_path)
                label_tmp.replace(label_path)
                marker.write_text(json.dumps({'source': signature, 'rows': rows}))
        self.images = np.load(image_path, mmap_mode='r')
        self.targets = np.load(label_path, mmap_mode='r')
        if self.images.shape != (len(self.targets), 3, 32, 32):
            raise ValueError('캐시의 이미지/라벨 개수가 맞지 않습니다.')
        if self.targets.min() < 0 or self.targets.max() >= 1000:
            raise ValueError('라벨은 0~999여야 합니다.')
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(np.array(self.images[index], copy=True)).float().div_(255)
        return (image - self.mean) / self.std, int(self.targets[index])


def load_images(root):
    if (root / 'data').is_dir() and any((root / 'data').glob('train-*.parquet')):
        train = CachedImageNet32(root, 'train')
        validation = CachedImageNet32(root, 'val')
        return train, validation, [f'class_{i:04d}' for i in range(1000)]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = datasets.ImageFolder(str(root / 'train'), transform=transform)
    validation = datasets.ImageFolder(str(root / 'val'), transform=transform)
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError('train과 val의 클래스 폴더 이름이 같아야 합니다.')
    if len(train.classes) < 2:
        raise ValueError('두 클래스 이상 필요합니다.')
    return train, validation, train.classes


def make_pairs(class_count, task_count, seed):
    if not 1 <= task_count <= class_count * (class_count - 1) // 2:
        raise ValueError('num_tasks가 중복 없는 클래스 쌍 개수를 초과합니다.')
    rng = random.Random(seed)
    pairs, used = [], set()
    while len(pairs) < task_count:
        pair = tuple(sorted(rng.sample(range(class_count), 2)))
        if pair not in used:
            used.add(pair)
            pairs.append(pair)
    return pairs


def parameter_count(widths):
    conv_cost = sum((left * 9 + 1) * right for left, right in zip((3, *widths[:-1]), widths))
    return conv_cost + (widths[-1] + 1) * 2


def allocate_widths(survivors, budget):
    def widths(scale):
        return [max(count, math.floor(count * scale)) for count in survivors]
    low, high = 1.0, 2.0
    if parameter_count(survivors) > budget:
        raise ValueError('생존 채널 수가 파라미터 예산을 초과합니다.')
    while parameter_count(widths(high)) <= budget:
        high *= 2
    for _ in range(70):
        middle = (low + high) / 2
        if parameter_count(widths(middle)) <= budget:
            low = middle
        else:
            high = middle
    return widths(low)


def mapped_tensor(old, new, output_indices, input_indices=None):
    """Conv/Linear의 생존 출력·입력 축을 새 텐서 앞부분에 복사."""
    if input_indices is None:
        new[:len(output_indices)] = old[output_indices]
    else:
        new[:len(output_indices), :len(input_indices)] = old[output_indices][:, input_indices]


def rejuvenate_channels(model, optimizer, utility, age, targets, budget):
    survivors = [[i for i in range(width) if i not in removed]
                 for width, removed in zip(model.widths, targets)]
    widths = allocate_widths([len(group) for group in survivors], budget)
    reference = next(model.parameters())
    rebuilt = ImageNetCNN(widths).to(device=reference.device, dtype=reference.dtype)
    rebuilt.train(model.training)
    mappings = {}
    with torch.no_grad():
        for i, (old, new) in enumerate(zip(model.convs, rebuilt.convs)):
            inputs = list(range(3)) if i == 0 else survivors[i - 1]
            outputs = survivors[i]
            new.weight[:len(outputs)].zero_()
            mapped_tensor(old.weight, new.weight, outputs, inputs)
            mapped_tensor(old.bias, new.bias, outputs)
            mappings[f'convs.{i}.weight'] = (outputs, inputs)
            mappings[f'convs.{i}.bias'] = (outputs, None)
        rebuilt.classifier.weight.zero_()
        mapped_tensor(model.classifier.weight, rebuilt.classifier.weight, [0, 1], survivors[-1])
        rebuilt.classifier.bias.copy_(model.classifier.bias)
        mappings['classifier.weight'] = ([0, 1], survivors[-1])
        mappings['classifier.bias'] = ([0, 1], None)

    new_optimizer = torch.optim.Adam(rebuilt.parameters(), **optimizer.defaults)
    old_parameters = dict(model.named_parameters())
    for name, param in rebuilt.named_parameters():
        old_state = optimizer.state.get(old_parameters[name], {})
        for key, value in old_state.items():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                buffer = value.new_zeros(param.shape)
                mapped_tensor(value, buffer, *mappings[name])
                new_optimizer.state[param][key] = buffer
            else:
                new_optimizer.state[param][key] = value.clone() if torch.is_tensor(value) else value
    new_utility = [reference.new_zeros(width) for width in widths]
    new_age = [torch.zeros(width, dtype=torch.long, device=reference.device) for width in widths]
    for i, keep in enumerate(survivors):
        new_utility[i][:len(keep)] = utility[i][keep]
        new_age[i][:len(keep)] = age[i][keep]
    return rebuilt, new_optimizer, new_utility, new_age


def clone_channels(model, optimizer, utility, age, targets, noise_std):
    with torch.no_grad():
        for i, removed in enumerate(targets):
            candidates = [j for j in range(model.widths[i]) if j not in removed]
            candidates.sort(key=lambda j: utility[i][j].item(), reverse=True)
            pool_size = max(len(removed), max(1, int(len(candidates) * 0.2)))
            sources = random.sample(candidates[:pool_size], min(len(removed), len(candidates)))
            # 소스를 중복 사용하지 않기 위해 실제 교체 수도 맞춘다.
            targets[i] = removed[:len(sources)]
            layer = model.convs[i]
            following = model.convs[i + 1] if i < 2 else model.classifier
            for target, source in zip(targets[i], sources):
                layer.weight[target].copy_(layer.weight[source])
                layer.bias[target].copy_(layer.bias[source])
                half = following.weight[:, source].clone() / 2
                following.weight[:, source] = half
                following.weight[:, target] = half
                noise = torch.randn_like(layer.weight[source]) * noise_std
                layer.weight[source].add_(noise)
                layer.weight[target].sub_(noise)
            changed = sorted(set(targets[i]) | set(sources))
            for param, axis in ((layer.weight, 0), (layer.bias, 0), (following.weight, 1)):
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    buffer = optimizer.state.get(param, {}).get(key)
                    if buffer is not None:
                        if axis == 0:
                            buffer[changed] = 0
                        else:
                            buffer[:, changed] = 0
            utility[i][targets[i]] = 0
            age[i][targets[i]] = 0
    return model, optimizer, utility, age


def reset_cbp_channels(model, optimizer, utility, age, targets):
    """CBP 변형: 입력을 초기 분포에서 뽑고 출력 연결을 0으로 만든다."""
    with torch.no_grad():
        for i, indices in enumerate(targets):
            if not indices:
                continue
            layer = model.convs[i]
            following = model.convs[i + 1] if i < 2 else model.classifier
            bound = 1 / math.sqrt(layer.in_channels * 9)
            layer.weight[indices] = torch.empty_like(layer.weight[indices]).uniform_(-bound, bound)
            layer.bias[indices] = torch.empty_like(layer.bias[indices]).uniform_(-bound, bound)
            following.weight[:, indices] = 0
            for param, axis in ((layer.weight, 0), (layer.bias, 0), (following.weight, 1)):
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    state = optimizer.state.get(param, {}).get(key)
                    if state is not None:
                        if axis == 0:
                            state[indices] = 0
                        else:
                            state[:, indices] = 0
            utility[i][indices] = 0
            age[i][indices] = 0
        # 다음 층의 새 행 초기화가 앞 층의 출력 0을 덮지 않도록 마지막에 확정.
        for i, indices in enumerate(targets):
            following = model.convs[i + 1] if i < 2 else model.classifier
            following.weight[:, indices] = 0
    return model, optimizer, utility, age


@torch.no_grad()
def detect_channels(model, loader, threshold):
    device = next(model.parameters()).device
    zero = [torch.zeros(width, dtype=torch.long) for width in model.widths]
    counts = [0] * len(model.widths)
    was_training = model.training
    model.eval()
    try:
        for images, _ in loader:
            model(images.to(device))
            for i, act in enumerate(model.activations):
                zero[i] += (act == 0).sum(dim=(0, 2, 3)).cpu()
                counts[i] += act.shape[0] * act.shape[2] * act.shape[3]
    finally:
        model.train(was_training)
    if not all(counts):
        raise ValueError('비활성 진단 데이터가 비었습니다.')
    ratios = [values.float() / count for values, count in zip(zero, counts)]
    dead = [torch.where(values >= threshold)[0].tolist() for values in ratios]
    stats = {'overall_dead_ratio': sum(map(len, dead)) / sum(model.widths),
             'by_layer': [{'dead_count': len(group), 'neuron_count': width,
                           'mean_zero_ratio': ratio.mean().item()}
                          for group, width, ratio in zip(dead, model.widths, ratios)]}
    return dead, stats


def save_progress(directory, model, optimizer, utility, age, log, cfg, task, epoch):
    """epoch 단위의 원자적 저장. 실행별 폴더로 덮어쓰기를 방지."""
    temporary = directory / 'checkpoint.tmp'
    torch.save({'model': model.state_dict(), 'widths': model.widths,
                'optimizer': optimizer.state_dict(), 'utility': utility, 'age': age,
                'task': task, 'epoch': epoch, 'global_step': log['global_step'],
                'config': cfg}, temporary)
    temporary.replace(directory / 'checkpoint.pt')
    temporary = directory / 'results.tmp'
    temporary.write_text(json.dumps({'config': cfg, 'results': log}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(directory / 'results.json')


def run(method, seed, cfg, train_base, val_base, class_names, output):
    random.seed(seed)
    torch.manual_seed(seed)
    device = net.set_device() if cfg['device'] == 'auto' else torch.device(cfg['device'])
    model = ImageNetCNN(cfg['widths']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()
    budget = parameter_count(model.widths)
    utility = [torch.zeros(width, device=device) for width in model.widths]
    age = [torch.zeros(width, dtype=torch.long, device=device) for width in model.widths]
    pairs = make_pairs(len(class_names), cfg['num_tasks'], seed)
    train_index = index_classes(train_base.targets, len(class_names))
    val_index = index_classes(val_base.targets, len(class_names))
    directory = output / f'{method}_seed{seed}'
    directory.mkdir(parents=True)
    selection = {
        'standard': 'none',
        'rejuvenation': 'mature_near_dead',
        'cbp': 'lowest_mature_utility',
        'ours': 'lowest_mature_utility',
        'ours_no_noise': 'lowest_mature_utility',
    }[method]
    log = {'target_selection': selection, 'step_acc': [], 'task_acc': [], 'dead_ratio': [], 'dead_ratio_by_layer': [],
           'regrow_steps': [], 'regrow_count': [], 'regrow_by_layer': [],
           'global_step': 0, 'epochs': [], 'regrow_events': [],
           'class_pairs': [[class_names[a], class_names[b]] for a, b in pairs],
           'parameter_budget': budget}

    def on_step():
        with torch.no_grad():
            for i, activation in enumerate(model.activations):
                following = model.convs[i + 1] if i < 2 else model.classifier
                weight_size = following.weight.abs().flatten(2).sum((0, 2)) if i < 2 else following.weight.abs().sum(0)
                contribution = activation.abs().mean((0, 2, 3)) * weight_size
                utility[i].mul_(cfg['beta']).add_(contribution, alpha=1 - cfg['beta'])
                age[i].add_(1)
            log['global_step'] += 1

    for task, pair in enumerate(pairs):
        train_set = ClassPair(train_base, pair, train_index)
        val_set = ClassPair(val_base, pair, val_index)
        # 배치 순서도 방법별 재생성 난수와 분리한다.
        generator = torch.Generator().manual_seed(seed * 1000000 + task)
        loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                            generator=generator, num_workers=cfg['workers'])
        diagnostic = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=cfg['workers'])
        validation = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=cfg['workers'])
        for epoch in range(cfg['epochs_per_task']):
            train_loss, train_acc = tn.train(model, loader, criterion, optimizer, log=log, on_step=on_step)
            dead, stats = detect_channels(model, diagnostic, cfg['near_dead_threshold'])
            log['dead_ratio'].append(stats['overall_dead_ratio'])
            log['dead_ratio_by_layer'].append(stats['by_layer'])
            targets = []
            if selection == 'mature_near_dead':
                targets = [[j for j in group if age[i][j].item() >= cfg['maturity']]
                           for i, group in enumerate(dead)]
            elif selection == 'lowest_mature_utility':
                for scores, ages in zip(utility, age):
                    mature = torch.where(ages >= cfg['maturity'])[0]
                    count = min(len(mature), max(1, int(len(mature) * cfg['p'])))
                    indices = torch.topk(scores[mature], k=count, largest=False).indices
                    targets.append(mature[indices].tolist())
            for i, group in enumerate(targets):
                if len(group) == model.widths[i]:
                    group.remove(utility[i].argmax().item())
            if method != 'standard' and any(targets):
                _, before = tn.evaluate(model, validation, criterion)
                old_widths = list(model.widths)
                if method == 'rejuvenation':
                    model, optimizer, utility, age = rejuvenate_channels(model, optimizer, utility, age, targets, budget)
                elif method == 'cbp':
                    model, optimizer, utility, age = reset_cbp_channels(model, optimizer, utility, age, targets)
                else:
                    noise = cfg['noise_std'] if method == 'ours' else 0.0
                    model, optimizer, utility, age = clone_channels(model, optimizer, utility, age, targets, noise)
                _, after = tn.evaluate(model, validation, criterion)
                log['regrow_steps'].append(log['global_step'])
                log['regrow_count'].append(sum(map(len, targets)))
                log['regrow_by_layer'].append(targets)
                log['regrow_events'].append({'step': log['global_step'], 'before': before,
                    'after': after, 'old_widths': old_widths, 'new_widths': list(model.widths)})
                print(f"\n[REGROW ACCURACY] {method} | Step {log['global_step']} | "
                      f"task {task + 1}/{cfg['num_tasks']} | epoch {epoch + 1}/{cfg['epochs_per_task']}", flush=True)
                print(f'Before : {before * 100:.2f}%\nAfter  : {after * 100:.2f}%\nChange : {(after - before) * 100:+.2f}%p', flush=True)
            _, accuracy = tn.evaluate(model, validation, criterion)
            log['epochs'].append({'task': task + 1, 'epoch': epoch + 1, 'train_loss': train_loss,
                                  'train_acc': train_acc, 'val_acc': accuracy, 'widths': list(model.widths)})
            if epoch == cfg['epochs_per_task'] - 1:
                log['task_acc'].append(accuracy)
            save_progress(directory, model, optimizer, utility, age, log, cfg, task, epoch)
            print(f"[{method} | seed {seed}] task {task + 1}/{cfg['num_tasks']} "
                  f"epoch {epoch + 1}/{cfg['epochs_per_task']} acc {accuracy * 100:.2f}%", flush=True)
    return log


def summarize_results(root):
    """원본 results.json을 읽어 비교 그래프와 분석용 CSV/JSON을 저장한다."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'imagenet-matplotlib'))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sources = sorted(path for path in root.glob('*/results.json')
                     if not path.parent.name.startswith('._'))
    if not sources:
        raise ValueError(f'{root} 아래에 방법별 results.json이 없습니다.')
    figure = Figure(figsize=(13, 9), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    summaries, tasks, epochs, events = [], [], [], []
    configs = {}
    for path in sources:
        payload = json.loads(path.read_text(encoding='utf-8'))
        log, cfg = payload['results'], payload['config']
        label = path.parent.name
        configs[label] = cfg
        accuracy = np.asarray(log['task_acc'], dtype=float) * 100
        dead = np.asarray(log['dead_ratio'], dtype=float) * 100
        changes = np.asarray([(event['after'] - event['before']) * 100
                              for event in log['regrow_events']], dtype=float)
        summary = {
            'run': label,
            'target_selection': log.get('target_selection', 'unrecorded'),
            'completed_tasks': len(accuracy),
            'completed_epochs': len(log['epochs']),
            'requested_tasks': cfg['num_tasks'],
            'epochs_per_task': cfg['epochs_per_task'],
            'complete': len(accuracy) == cfg['num_tasks'] and len(log['epochs']) == cfg['num_tasks'] * cfg['epochs_per_task'],
            'mean_task_end_accuracy_pct': float(accuracy.mean()) if len(accuracy) else None,
            'last_task_accuracy_pct': float(accuracy[-1]) if len(accuracy) else None,
            'mean_dead_ratio_pct': float(dead.mean()) if len(dead) else None,
            'last_dead_ratio_pct': float(dead[-1]) if len(dead) else None,
            'replacement_events': len(changes),
            'replaced_channels_total': sum(log['regrow_count']),
            'mean_immediate_change_pp': float(changes.mean()) if len(changes) else None,
            'worst_immediate_change_pp': float(changes.min()) if len(changes) else None,
            'accuracy_drop_events': int((changes < -1e-8).sum()),
            'source': str(path.resolve()),
        }
        summaries.append(summary)
        for index, value in enumerate(accuracy):
            tasks.append({'run': label, 'task': index + 1, 'accuracy_pct': float(value),
                          'class_pair': json.dumps(log['class_pairs'][index], ensure_ascii=False)})
        for index, entry in enumerate(log['epochs']):
            epochs.append({'run': label, 'task': entry['task'], 'epoch': entry['epoch'],
                           'train_loss': entry['train_loss'], 'train_accuracy_pct': entry['train_acc'] * 100,
                           'validation_accuracy_pct': entry['val_acc'] * 100,
                           'dead_ratio_pct': float(dead[index])})
        for event, count in zip(log['regrow_events'], log['regrow_count']):
            events.append({'run': label, 'step': event['step'], 'replaced_channels': count,
                           'before_pct': event['before'] * 100, 'after_pct': event['after'] * 100,
                           'change_pp': (event['after'] - event['before']) * 100})
        axes[0, 0].plot(np.arange(1, len(accuracy) + 1), accuracy, label=label, linewidth=1.5)
        axes[0, 1].plot(np.arange(1, len(dead) + 1), dead, label=label, linewidth=1, alpha=0.85)
        if len(changes):
            ordered = np.sort(changes)
            axes[1, 0].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100,
                            where='post', label=f'{label} (n={len(ordered)})')
    axes[0, 0].set(title='Validation accuracy at the end of each task', xlabel='Task', ylabel='Accuracy (%)', ylim=(0, 100))
    axes[0, 1].set(title='Inactive channel ratio before replacement', xlabel='Epoch across tasks', ylabel='Inactive channels (%)')
    axes[1, 0].set(title='Immediate accuracy change at replacement', xlabel='After - before (percentage points)', ylabel='Cumulative events (%)', ylim=(0, 102))
    axes[1, 0].axvline(0, color='gray', linewidth=1, linestyle='--')
    labels = [row['run'] for row in summaries]
    x = np.arange(len(labels))
    for offset, key, title in [(-0.18, 'mean_task_end_accuracy_pct', 'Mean of task-end accuracies'),
                               (0.18, 'last_task_accuracy_pct', 'Last task accuracy')]:
        values = [row[key] if row[key] is not None else np.nan for row in summaries]
        bars = axes[1, 1].bar(x + offset, values, width=0.36, label=title)
        axes[1, 1].bar_label(bars, fmt='%.1f', padding=3)
    axes[1, 1].set(title='Task-end accuracy summary', ylabel='Accuracy (%)', ylim=(0, 105), xticks=x, xticklabels=labels)
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    axes[1, 1].legend(loc='lower center', fontsize=8)
    figure.suptitle('Continual ImageNet32: recorded run comparison', fontsize=16)
    figure.savefig(root / 'comparison.png', dpi=180)
    figure.savefig(root / 'comparison.pdf')
    figure.clear()

    for filename, rows in [('summary.csv', summaries), ('task_accuracy.csv', tasks),
                            ('epoch_metrics.csv', epochs), ('replacement_events.csv', events)]:
        if rows:
            with (root / filename).open('w', newline='', encoding='utf-8-sig') as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = {
        'runs': summaries,
        'configs': configs,
        'definitions': {
            'mean_task_end_accuracy_pct': 'Mean of each task accuracy measured when that task finished; not final-model accuracy on all tasks.',
            'last_dead_ratio_pct': 'Measured before the last epoch replacement.',
            'immediate_change_pp': 'Same validation set evaluated immediately after minus before replacement.',
        },
        'limitations': [
            'One seed provides no estimate of variation across independent seeds.',
            'Replacement counts and channel identities may differ between runs. Older logs do not record target_selection.',
            'No statistical significance test or retrospective evaluation of all tasks is implied.',
        ],
    }
    (root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'그래프와 요약 저장 완료: {root.resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plot-only', type=Path, help='학습 없이 기존 실행 폴더의 그래프·요약 생성')
    parser.add_argument('--data-root', type=Path, help='클래스별 폴더 또는 data/*.parquet가 있는 ImageNet 경로')
    parser.add_argument('--methods', nargs='+', choices=['standard', 'ours', 'ours_no_noise', 'rejuvenation', 'cbp'],
                        default=['standard', 'ours', 'rejuvenation'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--num-tasks', type=int, default=3)
    parser.add_argument('--epochs-per-task', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--widths', nargs=3, type=int, default=[16, 32, 64])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--p', type=float, default=0.05, help='cbp, ours, ours_no_noise의 epoch당 성숙 채널 중 낮은 utility 교체 비율')
    parser.add_argument('--maturity', type=int, default=100)
    parser.add_argument('--noise-std', type=float, default=0.02)
    parser.add_argument('--near-dead-threshold', type=float, default=0.99)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('imagenet_runs'))
    parser.add_argument('--smoke-test', action='store_true', help='다운로드 없이 가상 데이터로 연결 검증')
    args = parser.parse_args()
    if args.plot_only is not None:
        summarize_results(args.plot_only)
        return
    if args.num_tasks < 1 or args.epochs_per_task < 1 or args.batch_size < 1 or min(args.widths) < 1:
        parser.error('과제 수, epoch, 배치 크기와 채널 수는 양수여야 합니다.')
    if args.workers < 0 or args.maturity < 0 or args.lr <= 0 or args.noise_std < 0:
        parser.error('workers/maturity/noise는 0 이상, lr은 양수여야 합니다.')
    if not 0 <= args.beta < 1 or not 0 < args.near_dead_threshold <= 1:
        parser.error('beta는 [0,1), near-dead-threshold는 (0,1]이어야 합니다.')
    if not 0 < args.p < 1:
        parser.error('p는 0보다 크고 1보다 작아야 합니다.')
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error('methods와 seeds는 중복할 수 없습니다.')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('seed는 0 이상 2**32 미만이어야 합니다.')
    if args.smoke_test:
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(91)
        train = TensorDataset(torch.randn(24, 3, 32, 32, generator=generator), torch.arange(24) % 4)
        validation = TensorDataset(torch.randn(12, 3, 32, 32, generator=generator), torch.arange(12) % 4)
        train.targets, validation.targets = train.tensors[1], validation.tensors[1]
        classes = ['synthetic_0', 'synthetic_1', 'synthetic_2', 'synthetic_3']
    else:
        if args.data_root is None:
            parser.error('--data-root에 준비된 ImageNet 경로를 지정하세요.')
        train, validation, classes = load_images(args.data_root)
    cfg = vars(args).copy()
    cfg['data_root'] = str(args.data_root) if args.data_root is not None else None
    cfg['output_dir'] = str(args.output_dir)
    if args.num_tasks > len(classes) * (len(classes) - 1) // 2:
        parser.error('사용 가능한 클래스 쌍보다 과제 수가 많습니다.')
    output = args.output_dir / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'결과 저장 위치: {output.resolve()}', flush=True)
    for method in args.methods:
        for seed in args.seeds:
            run(method, seed, cfg, train, validation, classes, output)
    summarize_results(output)
    print('전체 실험이 완료되었습니다.', flush=True)


if __name__ == '__main__':
    main()
