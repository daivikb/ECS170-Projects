#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "torch",
#     "scikit-learn",
# ]
# ///

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


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


class Dataset_Loader(dataset):
    def __init__(self, folder):
        super().__init__('Stage 2', 'Predefined training and testing sets')
        self.folder = Path(folder)

    def load(self):
        data = {}
        for split, count in [('train', 60000), ('test', 10000)]:
            rows = np.loadtxt(self.folder / f'{split}.csv', delimiter=',', ndmin=2)
            if rows.shape != (count, 785):
                raise ValueError(f'{split}.csv must have {count} rows and 785 columns')
            if not np.isfinite(rows).all() or not np.equal(rows, np.floor(rows)).all():
                raise ValueError('Labels and features must be finite integers')
            if np.any((rows[:, 0] < 0) | (rows[:, 0] > 9)):
                raise ValueError('Labels must be between 0 and 9')
            if np.any((rows[:, 1:] < 0) | (rows[:, 1:] > 255)):
                raise ValueError('Features must be between 0 and 255')
            data[split] = (torch.from_numpy(rows[:, 1:].astype(np.float32) / 255),
                           torch.from_numpy(rows[:, 0].astype(np.int64)))
        return data


class Method_MLP(method, nn.Module):
    def __init__(self, config):
        method.__init__(self, config['name'], 'Multilayer perceptron')
        nn.Module.__init__(self)
        torch.manual_seed(config['seed'])
        self.config = config
        layers = []
        width = 784
        for size in config['hidden']:
            layer = nn.Linear(width, size)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.ReLU()])
            if config['dropout']:
                layers.append(nn.Dropout(config['dropout']))
            width = size
        output = nn.Linear(width, 10)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        self.network = nn.Sequential(*layers, output)

    def forward(self, x):
        return self.network(x)

    def loss(self, logits, labels):
        if self.config['loss'] == 'mse':
            return F.mse_loss(logits.softmax(1), F.one_hot(labels, 10).float())
        return F.cross_entropy(logits, labels)

    @torch.inference_mode()
    def snapshot(self, x, y):
        self.eval()
        loss = cross_entropy = correct = 0
        for start in range(0, len(y), 2048):
            xb, yb = x[start:start + 2048], y[start:start + 2048]
            logits = self(xb)
            loss += self.loss(logits, yb).item() * len(yb)
            cross_entropy += F.cross_entropy(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        return loss / len(y), cross_entropy / len(y), correct / len(y)

    def fit(self, x, y, x_eval=None, y_eval=None):
        c = self.config
        optimizers = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
                      'sgd': torch.optim.SGD}
        options = {'momentum': 0.9} if c['optimizer'] == 'sgd' else {}
        optimizer = optimizers[c['optimizer']](self.parameters(), lr=c['lr'],
                                               weight_decay=c['weight_decay'], **options)
        generator = torch.Generator().manual_seed(c['seed'])
        self.history = []
        for epoch in range(1, 21):
            self.train()
            for indices in torch.randperm(len(y), generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss(self(x[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
            loss, cross_entropy, accuracy = self.snapshot(x, y)
            # Test metrics are recorded for plotting only; they never influence training.
            test_loss, test_ce, test_acc = self.snapshot(x_eval, y_eval)
            self.history.append([epoch, loss, cross_entropy, accuracy,
                                 test_loss, test_ce, test_acc])
            print(f"{c['name']} seed {c['seed']} epoch {epoch:2}: train loss={loss:.6f}, "
                  f"train acc={accuracy:.2%}, test CE={test_ce:.6f}, test acc={test_acc:.2%}",
                  flush=True)

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        return torch.cat([self(batch).argmax(1) for batch in x.split(2048)]).numpy()

    def run(self, data):
        self.fit(*data['train'], *data['test'])
        return self.predict(data['test'][0])


class Evaluate_Metrics(evaluate):
    def evaluate(self, truth, predictions):
        scores = {'accuracy': float(accuracy_score(truth, predictions))}
        for average in ['macro', 'weighted', 'micro']:
            p, r, f, _ = precision_recall_fscore_support(
                truth, predictions, labels=range(10), average=average, zero_division=0)
            scores[average] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}
        p, r, f, support = precision_recall_fscore_support(
            truth, predictions, labels=range(10), zero_division=0)
        scores['per_class'] = [{'label': i, 'precision': float(p[i]), 'recall': float(r[i]),
                                'f1': float(f[i]), 'support': int(support[i])} for i in range(10)]
        scores['confusion_matrix'] = confusion_matrix(truth, predictions, labels=range(10)).tolist()
        return scores


class Result_Saver(result):
    def __init__(self, folder):
        super().__init__('Stage 2 results')
        self.folder = Path(folder)

    def save(self, model, truth, predictions, scores):
        folder = self.folder / model.config['name'] / f"seed_{model.config['seed']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'metrics.json').write_text(json.dumps(scores, indent=2))
        np.savetxt(folder / 'history.csv', model.history, delimiter=',',
                   header='epoch,loss,cross_entropy,accuracy,test_loss,test_cross_entropy,test_accuracy',
                   comments='')
        np.savetxt(folder / 'predictions.csv', np.column_stack([truth, predictions]),
                   delimiter=',', fmt='%d', header='true_label,predicted_label', comments='')
        torch.save({'config': model.config, 'state_dict': model.state_dict()}, folder / 'model.pt')

    def load(self, name, seed):
        return json.loads((self.folder / name / f'seed_{seed}' / 'metrics.json').read_text())


class Setting_Predefined_Split(setting):
    def load_run_save_evaluate(self, data=None):
        if data is None:
            data = self.dataset.load()
        predictions = self.method.run(data)
        truth = data['test'][1].numpy()
        scores = self.evaluate.evaluate(truth, predictions)
        self.result.save(self.method, truth, predictions, scores)
        return scores


def load_histories(folder, name, seeds):
    return np.stack([np.loadtxt(folder / name / f'seed_{seed}' / 'history.csv',
                                delimiter=',', skiprows=1) for seed in seeds])


def plot_results(folder, results, seeds):
    folder = Path(folder)
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    epochs = np.arange(1, 21)
    colors = {name: f'C{i}' for i, name in enumerate(results)}
    histories = {name: load_histories(folder, name, seeds) for name in results}

    # Train vs. test curves: line = mean over seeds, band = +/- 1 std over seeds.
    for columns, filename, ylabel, scale, log in [
        ((2, 5), '01_loss_vs_epoch', 'Cross-entropy', 1, True),
        ((3, 6), '02_accuracy_vs_epoch', 'Accuracy (%)', 100, False),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharey=log)
        for ax, column, title in zip(axes, columns, ['Training set', 'Test set']):
            for name, h in histories.items():
                values = h[:, :, column] * scale
                mean = values.mean(0)
                ax.plot(epochs, mean, label=name, color=colors[name])
                if len(seeds) > 1:
                    ax.fill_between(epochs, mean - values.std(0, ddof=1),
                                    mean + values.std(0, ddof=1), color=colors[name], alpha=0.2)
            ax.set(xlabel='Epoch', ylabel=ylabel, xticks=[1, 5, 10, 15, 20],
                   title=f'{title} (mean over {len(seeds)} seeds)')
            if log:
                ax.set_yscale('log')
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.savefig(figures / f'{filename}.png', dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    positions = np.arange(len(results))
    for offset, label, key in [(-0.18, 'Accuracy', lambda s: s['accuracy']),
                               (0.18, 'Macro F1', lambda s: s['macro']['f1'])]:
        values = np.array([[key(s) * 100 for s in runs] for runs in results.values()])
        errors = values.std(1, ddof=1) if len(seeds) > 1 else None
        bars = ax.bar(positions + offset, values.mean(1), 0.36, yerr=errors,
                      capsize=3, label=label)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=8)
    ax.set(xticks=positions, xticklabels=[name[0] for name in results],
           xlabel='Experiment', ylabel=f'Test score (%), mean +/- std over {len(seeds)} seeds',
           ylim=(0, 108))
    ax.legend(loc='lower right')
    fig.savefig(figures / '03_test_performance_comparison.png', dpi=220)
    plt.close(fig)

    primary = results['F_regularized'][0]  # first seed
    matrix = np.array(primary['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.imshow(matrix, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted label',
           ylabel='True label', title=f'F_regularized: test confusion matrix (seed {seeds[0]})')
    fig.savefig(figures / '04_confusion_matrix.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for metric in ['precision', 'recall', 'f1']:
        ax.plot(range(10), [c[metric] * 100 for c in primary['per_class']], marker='o', label=metric)
    ax.set(xticks=range(10), xlabel='Class label', ylabel='Test score (%)', ylim=(95, 100),
           title=f'F_regularized: per-class metrics (seed {seeds[0]})')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / '05_per_class_metrics.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2), layout='constrained')
    ax.axis('off')
    labels = ['784 inputs\nPixels / 255', '256 units\nReLU + dropout 0.2',
              '128 units\nReLU + dropout 0.2', '10 logits\nArgmax']
    for i, label in enumerate(labels):
        x = 0.12 + i * 0.25
        ax.text(x, 0.55, label, ha='center', va='center',
                bbox={'boxstyle': 'round,pad=0.6', 'facecolor': 'aliceblue'})
        if i < 3:
            ax.annotate('', xy=(x + 0.16, 0.55), xytext=(x + 0.09, 0.55),
                        arrowprops={'arrowstyle': '->'})
    ax.set_title('Primary model F: 784 - 256 - 128 - 10')
    fig.savefig(figures / 'model_architecture.png', dpi=220)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    folders = [root / 'data', root / 'stage_2_data', root / 'data/stage_2_data',
               root.parent / 'data/stage_2_data']
    default_data = next((p for p in folders if (p / 'train.csv').is_file()
                         and (p / 'test.csv').is_file()), root / 'data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=default_data)
    parser.add_argument('--output-dir', type=Path, default=root / 'results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    if not all((args.data_dir / name).is_file() for name in ['train.csv', 'test.csv']):
        parser.error('Dataset not found. Use --data-dir followed by the folder containing train.csv and test.csv')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    loader = Dataset_Loader(args.data_dir)
    data = loader.load()
    base = dict(hidden=[256, 128], loss='cross_entropy', optimizer='adam', lr=0.001,
                dropout=0, weight_decay=0)
    configs = [dict(base, name='A_baseline', hidden=[128]),
               dict(base, name='B_two_hidden'),
               dict(base, name='C_three_hidden', hidden=[256, 128, 64]),
               dict(base, name='D_sgd', optimizer='sgd', lr=0.1),
               dict(base, name='E_mse', loss='mse'),
               dict(base, name='F_regularized', optimizer='adamw', dropout=0.2, weight_decay=0.0001)]
    results = {}
    for config in configs:
        results[config['name']] = []
        for seed in args.seeds:
            run_config = dict(config, seed=seed)
            experiment = Setting_Predefined_Split('Predefined split')
            experiment.prepare(loader, Method_MLP(run_config), Result_Saver(args.output_dir),
                               Evaluate_Metrics())
            scores = experiment.load_run_save_evaluate(data)
            results[config['name']].append(scores)
            print(f"{config['name']} seed {seed}: test accuracy = {scores['accuracy']:.2%}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def mean_std(values):
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    with (args.output_dir / 'comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['experiment', 'n_seeds', 'accuracy_mean', 'accuracy_std',
                         'macro_precision_mean', 'macro_precision_std', 'macro_recall_mean',
                         'macro_recall_std', 'macro_f1_mean', 'macro_f1_std'])
        for name, runs in results.items():
            row = [name, len(runs)]
            for get in [lambda s: s['accuracy'], lambda s: s['macro']['precision'],
                        lambda s: s['macro']['recall'], lambda s: s['macro']['f1']]:
                row.extend(mean_std([get(s) for s in runs]))
            writer.writerow(row)
    plot_results(args.output_dir, results, args.seeds)


if __name__ == '__main__':
    main()
