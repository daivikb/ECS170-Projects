import argparse
import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F


# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

class dataset:
    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_descrition = dDescription

    def load(self):
        raise NotImplementedError


class method:
    def __init__(self, mName=None, mDescription=None):
        self.method_name = mName
        self.method_description = mDescription

    def run(self):
        raise NotImplementedError


class evaluate:
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    def evaluate(self):
        raise NotImplementedError


class result:
    def __init__(self, rName=None, rType=None):
        self.result_name = rName
        self.result_description = rType

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class setting:
    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def load_run_save_evaluate(self):
        raise NotImplementedError


DATASETS = {
    'MNIST': dict(shape=(28, 28), counts=(60000, 10000), classes=10, offset=0, epochs=10),
    'ORL': dict(shape=(112, 92, 3), counts=(360, 40), classes=40, offset=1, epochs=40),
    'CIFAR': dict(shape=(32, 32, 3), counts=(50000, 10000), classes=10, offset=0, epochs=10),
}
VARIANTS = {
    'A_baseline': dict(depth=2, pool='max', loss='cross_entropy'),
    'B_deeper': dict(depth=3, pool='max', loss='cross_entropy'),
    'C_average_pool': dict(depth=2, pool='average', loss='cross_entropy'),
    'D_mse': dict(depth=2, pool='max', loss='mse'),
}
HISTORY_FIELDS = ['epoch', 'train_loss', 'train_cross_entropy', 'train_accuracy',
                  'test_loss', 'test_cross_entropy', 'test_accuracy', 'seconds']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Dataset_Loader(dataset):
    def __init__(self, folder, name):
        super().__init__(name, 'Instructor-provided image partitions')
        self.path = Path(folder) / name
        self.spec = DATASETS[name]

    def load(self):
        with self.path.open('rb') as stream:
            raw = pickle.load(stream)
        self.digest = sha256(self.path)
        data = {}
        for split, count in zip(['train', 'test'], self.spec['counts']):
            items = raw[split]
            if len(items) != count:
                raise ValueError(f'{self.dataset_name} {split}: expected {count} images')
            images = np.stack([item['image'] for item in items])
            labels = np.asarray([item['label'] for item in items])
            if images.shape != (count, *self.spec['shape']):
                raise ValueError(f'Unexpected image shape: {images.shape}')
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
                raise ValueError('Pixel values must be finite and between 0 and 255')
            if labels.shape != (count,) or not np.isfinite(labels).all():
                raise ValueError('Each image must have one finite label')
            if not np.equal(labels, np.floor(labels)).all():
                raise ValueError('Labels must be integers')
            labels = labels.astype(np.int64) - self.spec['offset']
            if set(labels.tolist()) != set(range(self.spec['classes'])):
                raise ValueError('Unexpected or missing class labels')
            if self.dataset_name == 'MNIST':
                images = images[:, None, :, :]
            elif self.dataset_name == 'ORL':
                if not (np.array_equal(images[..., 0], images[..., 1])
                        and np.array_equal(images[..., 0], images[..., 2])):
                    raise ValueError('Expected identical ORL RGB channels')
                images = images[..., 0][:, None, :, :]
            else:
                images = images.transpose(0, 3, 1, 2)
            pixels = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))
            data[split] = (pixels.div_(255).contiguous(memory_format=torch.channels_last),
                           torch.from_numpy(labels))
        return data


class Method_CNN(method, nn.Module):
    def __init__(self, config):
        method.__init__(self, config['name'], 'Convolutional neural network')
        nn.Module.__init__(self)
        torch.manual_seed(config['seed'])
        self.config = config
        self.spec = DATASETS[config['dataset']]
        self.device = torch.device(config['device'])
        channels = 3 if config['dataset'] == 'CIFAR' else 1
        height, width = self.spec['shape'][:2]
        layers = []
        if config['dataset'] == 'ORL':
            layers.append(nn.AvgPool2d(2))
            height, width = height // 2, width // 2
        pool = nn.MaxPool2d if config['pool'] == 'max' else nn.AvgPool2d
        for index, output_channels in enumerate([8, 16]):
            layers.extend([nn.Conv2d(channels, output_channels, 3, padding=1), nn.ReLU()])
            if index == 1 and config['depth'] == 3:
                layers.extend([nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.ReLU()])
            layers.append(pool(2))
            channels = output_channels
            height, width = height // 2, width // 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels * height * width, 128),
                                        nn.ReLU(), nn.Linear(128, self.spec['classes']))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        self.to(self.device, memory_format=torch.channels_last)

    def forward(self, x):
        return self.classifier(self.features(x))

    def loss(self, logits, labels):
        if self.config['loss'] == 'mse':
            return F.mse_loss(logits.softmax(1), F.one_hot(labels, self.spec['classes']).float())
        return F.cross_entropy(logits, labels)

    @torch.inference_mode()
    def snapshot(self, x, y):
        self.eval()
        objective = cross_entropy = correct = 0
        for start in range(0, len(y), self.config['batch_size']):
            xb, yb = x[start:start + self.config['batch_size']].to(self.device), y[start:start + self.config['batch_size']].to(self.device)
            logits = self(xb)
            objective += self.loss(logits, yb).item() * len(yb)
            cross_entropy += F.cross_entropy(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        return objective / len(y), cross_entropy / len(y), correct / len(y)

    def fit(self, x, y, test_x, test_y):
        c = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=c['lr'])
        generator = torch.Generator().manual_seed(c['seed'])
        self.history = []
        for epoch in range(1, c['epochs'] + 1):
            started = time.perf_counter()
            self.train()
            for indices in torch.randperm(len(y), generator=generator).split(c['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                logits = self(x[indices].to(self.device))
                loss = self.loss(logits, y[indices].to(self.device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
            train_scores = self.snapshot(x, y)
            test_scores = self.snapshot(test_x, test_y)
            elapsed = time.perf_counter() - started
            self.history.append([epoch, *train_scores, *test_scores, elapsed])
            print(f"{c['dataset']} {c['name']} seed {c['seed']} epoch {epoch}/{c['epochs']}: "
                  f"train CE={train_scores[1]:.4f}, train acc={train_scores[2]:.2%}, "
                  f"test CE={test_scores[1]:.4f}, test acc={test_scores[2]:.2%} ({elapsed:.1f}s)", flush=True)

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        return torch.cat([self(batch.to(self.device)).argmax(1).cpu()
                          for batch in x.split(self.config['batch_size'])]).numpy()

    def run(self, data):
        self.fit(*data['train'], *data['test'])
        return self.predict(data['test'][0])


class Evaluate_Metrics(evaluate):
    def __init__(self, name):
        super().__init__('Multiclass metrics')
        spec = DATASETS[name]
        self.labels = list(range(spec['offset'], spec['offset'] + spec['classes']))

    def evaluate(self, truth, predictions):
        scores = {'accuracy': float(accuracy_score(truth, predictions))}
        for average in ['macro', 'weighted', 'micro']:
            p, r, f, _ = precision_recall_fscore_support(
                truth, predictions, labels=self.labels, average=average, zero_division=0)
            scores[average] = dict(precision=float(p), recall=float(r), f1=float(f))
        p, r, f, support = precision_recall_fscore_support(
            truth, predictions, labels=self.labels, zero_division=0)
        scores['per_class'] = [dict(label=label, precision=float(p[i]), recall=float(r[i]),
                                    f1=float(f[i]), support=int(support[i]))
                               for i, label in enumerate(self.labels)]
        scores['confusion_matrix'] = confusion_matrix(truth, predictions, labels=self.labels).tolist()
        return scores


class Result_Saver(result):
    def __init__(self, folder, config):
        super().__init__('Stage 3 results')
        self.folder = Path(folder) / config['dataset'] / config['name'] / f"seed_{config['seed']}"

    def save(self, model, truth, predictions, scores, digest):
        self.folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.folder / 'history.csv', model.history, delimiter=',',
                   header=','.join(HISTORY_FIELDS), comments='')
        np.savetxt(self.folder / 'predictions.csv', np.column_stack([truth, predictions]),
                   delimiter=',', fmt='%d', header='true_label,predicted_label', comments='')
        torch.save(dict(config=model.config, state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                        dataset_sha256=digest), self.folder / 'model.pt')
        scores.update(config=model.config, dataset_sha256=digest,
                      parameters=sum(p.numel() for p in model.parameters()),
                      final_train=dict(zip(['loss', 'cross_entropy', 'accuracy'], model.history[-1][1:4])),
                      final_test=dict(zip(['loss', 'cross_entropy', 'accuracy'], model.history[-1][4:7])),
                      training_seconds=sum(row[-1] for row in model.history),
                      environment=dict(python=platform.python_version(), torch=torch.__version__,
                                       numpy=np.__version__, sklearn=sklearn.__version__, platform=platform.platform()),
                      source_sha256=sha256(__file__))
        (self.folder / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')

    def load(self):
        return json.loads((self.folder / 'metrics.json').read_text())


class Setting_Predefined_Split(setting):
    def load_run_save_evaluate(self, data=None):
        if data is None:
            data = self.dataset.load()
        offset = self.dataset.spec['offset']
        predictions = self.method.run(data) + offset
        truth = data['test'][1].numpy() + offset
        scores = self.evaluate.evaluate(truth, predictions)
        self.result.save(self.method, truth, predictions, scores, self.dataset.digest)
        return scores


def make_config(name, variant, seed, device='cpu', epochs=None, threads=4):
    return dict(dataset=name, name=variant, seed=seed, device=device, threads=threads,
                epochs=epochs or DATASETS[name]['epochs'], batch_size=32 if name == 'ORL' else 256,
                lr=0.001, **VARIANTS[variant])


def mean_sd(values):
    values = np.asarray(values)
    return values.mean(axis=0), values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1:])


def summarize(folder, name, variants, seeds):
    folder = Path(folder) / name
    figures = folder / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    scores, histories = {}, {}
    rows = []
    for variant in variants:
        scores[variant] = [json.loads((folder / variant / f'seed_{s}' / 'metrics.json').read_text()) for s in seeds]
        first = scores[variant][0]
        for seed, score in zip(seeds, scores[variant]):
            expected = dict(first['config'], seed=seed)
            if score['config'] != expected or score['config']['dataset'] != name or score['config']['name'] != variant:
                raise ValueError('Cannot combine runs with different experimental settings')
            if any(score[key] != first[key] for key in ['source_sha256', 'dataset_sha256']):
                raise ValueError('Cannot combine runs from different code or data')
        histories[variant] = np.stack([np.loadtxt(folder / variant / f'seed_{s}' / 'history.csv',
                                                delimiter=',', skiprows=1, ndmin=2) for s in seeds])
        row = dict(dataset=name, experiment=variant, seeds=';'.join(map(str, seeds)))
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            values = [s['accuracy'] if key == 'accuracy' else s['macro'][key] for s in scores[variant]]
            row[key + '_mean'], row[key + '_sd'] = map(float, mean_sd(values))
        rows.append(row)
    with (folder / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

    for column, filename, ylabel, scale in [(2, '01_loss_vs_epoch', 'Cross-entropy', 1),
                                           (3, '02_accuracy_vs_epoch', 'Accuracy (%)', 100)]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for i, variant in enumerate(variants):
            history = histories[variant]
            for ax, index in zip(axes, [column, column + 3]):
                average, sd = mean_sd(history[:, :, index] * scale)
                epochs = history[0, :, 0]
                ax.plot(epochs, average, label=variant, color=f'C{i}')
                ax.fill_between(epochs, np.maximum(average - sd, 0),
                                np.minimum(average + sd, 100) if scale == 100 else average + sd,
                                color=f'C{i}', alpha=0.15)
        for ax, title in zip(axes, ['Training (evaluation mode)', 'Test (diagnostic only)']):
            ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        fig.suptitle(f'{name}: mean ± sample SD across {len(seeds)} seeds')
        fig.savefig(figures / f'{filename}.png', dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    positions = np.arange(len(rows))
    for offset, metric, label in [(-0.18, 'accuracy', 'Accuracy'), (0.18, 'f1', 'Macro F1')]:
        bars = ax.bar(positions + offset, [r[metric + '_mean'] * 100 for r in rows], 0.36,
                      yerr=[r[metric + '_sd'] * 100 for r in rows], capsize=3, label=label)
        ax.bar_label(bars, fmt='%.2f', padding=5, fontsize=8)
    ax.set(xticks=positions, xticklabels=list(variants), ylabel='Test score (%)', ylim=(0, 110),
           title=f'{name}: final test scores, mean ± sample SD')
    ax.legend(loc='lower right')
    fig.savefig(figures / '03_test_comparison.png', dpi=180)
    plt.close(fig)

    count = len(variants)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), layout='constrained', squeeze=False)
    labels = [c['label'] for c in scores[variants[0]][0]['per_class']]
    for ax, variant in zip(axes[0], variants):
        matrix = np.mean([s['confusion_matrix'] for s in scores[variant]], axis=0)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        plot = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        step = 5 if len(labels) > 10 else 1
        ticks = list(range(0, len(labels), step))
        ax.set(xticks=ticks, yticks=ticks, xticklabels=labels[::step], yticklabels=labels[::step],
               xlabel='Predicted label', ylabel='True label', title=variant)
        if len(labels) == 10:
            for i in range(10):
                for j in range(10):
                    if normalized[i, j] >= 0.01:
                        ax.text(j, i, f'{normalized[i, j]:.0%}', ha='center', va='center', fontsize=7,
                                color='white' if normalized[i, j] > 0.5 else 'black')
    fig.colorbar(plot, ax=list(axes[0]), label='Fraction of true class', shrink=0.75)
    fig.suptitle(f'{name}: confusion matrices averaged over {len(seeds)} runs, row normalized')
    fig.savefig(figures / '04_confusion_matrices.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
        for i, variant in enumerate(variants):
            values = [[c[metric] * 100 for c in s['per_class']] for s in scores[variant]]
            average, sd = mean_sd(values)
            ax.plot(labels, average, marker='.' if len(labels) > 10 else 'o', markersize=3,
                    label=variant, color=f'C{i}', linewidth=1)
            ax.fill_between(labels, np.maximum(average - sd, 0), np.minimum(average + sd, 100),
                            color=f'C{i}', alpha=0.12)
        ax.set(xlabel='Original class label', ylabel=f'{metric.capitalize()} (%)', ylim=(-2, 102))
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{name}: per-class metrics, mean ± sample SD')
    fig.savefig(figures / '05_per_class_metrics.png', dpi=180)
    plt.close(fig)
    return rows


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Stage 3 CNN experiments on the supplied image datasets')
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--output-dir', type=Path, default=root / 'results')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--configs', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('Threads and epochs must be positive')
    for values in [args.datasets, args.configs, args.seeds]:
        if len(values) != len(set(values)):
            parser.error('Datasets, configurations, and seeds must not contain duplicates')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('Seeds must be between 0 and 2**32 - 1')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    for name in args.datasets:
        if not args.plots_only:
            if not (args.data_dir / name).is_file():
                parser.error(f'Missing {args.data_dir / name}. Extract the supplied stage_3_data.zip into data/')
            loader = Dataset_Loader(args.data_dir, name)
            data = loader.load()
            for variant in args.configs:
                for seed in args.seeds:
                    config = make_config(name, variant, seed, args.device, args.epochs, args.threads)
                    saver = Result_Saver(args.output_dir, config)
                    if args.resume and (saver.folder / 'metrics.json').is_file():
                        previous = saver.load()
                        if previous['config'] != config or previous['dataset_sha256'] != loader.digest:
                            raise ValueError(f'Configuration/data mismatch in {saver.folder}; use a new output directory')
                        if previous['source_sha256'] != sha256(__file__):
                            raise ValueError('Source changed; use a new output directory for the new experiment')
                        if all((saver.folder / p).is_file() for p in ['model.pt', 'history.csv', 'predictions.csv']):
                            print(f'Skipping completed {name} {variant} seed {seed}', flush=True)
                            continue
                    experiment = Setting_Predefined_Split('Supplied train/test split')
                    experiment.prepare(loader, Method_CNN(config), saver, Evaluate_Metrics(name))
                    experiment.load_run_save_evaluate(data)
            del data
        rows = summarize(args.output_dir, name, args.configs, args.seeds)
        for row in rows:
            print(f"{name} {row['experiment']}: accuracy {row['accuracy_mean']:.2%} "
                  f"± {row['accuracy_sd'] * 100:.3f} percentage points", flush=True)


if __name__ == '__main__':
    main()
