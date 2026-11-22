#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy==1.26.4",
#     "scipy==1.13.1",
#     "scikit-learn==1.5.2",
#     "torch==2.2.2",
#     "matplotlib==3.10.9",
# ]
# ///
"""ECS 170 Stage 4: recurrent sentiment classification and joke generation.

The five starter-template roles are kept together, following the Stage 3 layout.
Run with --help for training, plotting, and three-word generation commands.
"""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import platform
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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


SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']
CELLS = ['RNN', 'LSTM', 'GRU']
TASKS = ['classification', 'generation']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', html.unescape(text)).lower().replace('’', "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*|[.,!?;:]", text)


def words(tokens):
    return [t for t in tokens if re.search('[a-z0-9]', t)]


def detokenize(tokens):
    return re.sub(r'\s+([.,!?;:])', r'\1', ' '.join(tokens))


def vocabulary(documents, limit):
    counts = Counter(t for doc in documents for t in doc)
    ordered = sorted(counts, key=lambda t: (-counts[t], t))
    return SPECIAL + ordered[:limit - len(SPECIAL)]


def encode(tokens, vocab, length=None):
    ids = [vocab.get(t, 1) for t in tokens] or [1]
    return ids[:length] if length else ids


def pad_rows(rows):
    result = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    for i, row in enumerate(rows):
        result[i, :len(row)] = torch.tensor(row)
    return result


def batches(data, size, generator=None):
    x, y = data
    indices = torch.randperm(len(x), generator=generator) if generator else torch.arange(len(x))
    for ix in indices.split(size):
        xb = x[ix]
        # Shorten each batch without changing real tokens or the recurrent context.
        width = int(xb.ne(0).sum(1).max())
        yield xb[:, :width], y[ix, :width] if y.ndim == 2 else y[ix]


class Dataset_Loader(dataset):
    """Load the instructor data with fixed splits and training-only vocabularies."""

    def __init__(self, folder, task, max_length=160, vocab_size=12000):
        super().__init__(task, 'Instructor text data with training-only vocabulary')
        self.folder, self.task = Path(folder), task
        self.max_length, self.vocab_size = max_length, vocab_size

    def load(self):
        digest = hashlib.sha256()
        if self.task == 'classification':
            docs, labels, paths = {}, {}, {}
            for split in ['train', 'test']:
                docs[split], labels[split], paths[split] = [], [], []
                for label, name in enumerate(['neg', 'pos']):
                    files = sorted((self.folder / 'text_classification' / split / name).glob('*.txt'))
                    if len(files) != 12500:
                        raise ValueError(f'Expected 12500 reviews in {split}/{name}, got {len(files)}')
                    for path in files:
                        raw = path.read_bytes()
                        relative = str(path.relative_to(self.folder))
                        digest.update(relative.encode() + b'\0' + raw + b'\0')
                        docs[split].append(tokenize(raw.decode('utf-8')))
                        labels[split].append(label)
                        paths[split].append(relative)
            train, valid = train_test_split(np.arange(25000), test_size=0.1, random_state=170,
                                            stratify=labels['train'])
            self.vocab = vocabulary([docs['train'][i] for i in train], self.vocab_size)
            mapping = {t: i for i, t in enumerate(self.vocab)}
            data = {}
            manifest = {}
            for split, source, indices in [('train', 'train', train), ('validation', 'train', valid),
                                            ('test', 'test', np.arange(25000))]:
                selected = [docs[source][i] for i in indices]
                data[split] = (pad_rows([encode(d, mapping, self.max_length) for d in selected]),
                               torch.tensor([labels[source][i] for i in indices]))
                manifest[split] = [paths[source][i] for i in indices]
            self.manifest = manifest
            self.metadata = dict(counts={s: len(x) for s, (x, _) in data.items()},
                                 max_length=self.max_length, vocabulary_size=len(self.vocab),
                                 train_truncated_fraction=float(np.mean([len(docs['train'][i]) > self.max_length for i in train])),
                                 split_seed=170, label_names=['negative', 'positive'])
        else:
            path = self.folder / 'text_generation' / 'data'
            digest.update(path.read_bytes())
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1622 or set(rows[0]) != {'ID', 'Joke'}:
                raise ValueError('Expected the supplied 1622-row ID,Joke CSV')
            # Keep identical normalized jokes in one partition; remove duplicate training examples.
            unique = {}
            for row in rows:
                tokens = tokenize(row['Joke'])
                if len(words(tokens)) >= 3:
                    unique.setdefault(tuple(tokens), row['ID'])
            keys = sorted(unique)
            train, valid = train_test_split(np.arange(len(keys)), test_size=0.1, random_state=170)
            self.train_tokens = [list(keys[i]) for i in train]
            self.valid_tokens = [list(keys[i]) for i in valid]
            self.vocab = vocabulary(self.train_tokens, 8000)
            mapping = {t: i for i, t in enumerate(self.vocab)}
            data, manifest = {}, {}
            for split, indices in [('train', train), ('validation', valid)]:
                sequences = [[2] + encode(list(keys[i]), mapping) + [3] for i in indices]
                data[split] = (pad_rows([s[:-1] for s in sequences]), pad_rows([s[1:] for s in sequences]))
                manifest[split] = [unique[keys[i]] for i in indices]
            self.manifest = manifest
            valid_tokens = [t for doc in self.valid_tokens for t in doc]
            self.metadata = dict(original_jokes=len(rows), unique_jokes=len(keys),
                                 counts={s: len(x) for s, (x, _) in data.items()},
                                 vocabulary_size=len(self.vocab), split_seed=170,
                                 validation_unknown_fraction=sum(t not in mapping for t in valid_tokens)/len(valid_tokens))
        self.digest = digest.hexdigest()
        self.metadata['dataset_sha256'] = self.digest
        return data


class Method_RNN(method, nn.Module):
    """A separate RNN, LSTM, or GRU for either classification or next-token prediction."""

    def __init__(self, config):
        method.__init__(self, config['cell'], 'Recurrent text model')
        nn.Module.__init__(self)
        self.config = config
        torch.manual_seed(config['seed'])
        self.embedding = nn.Embedding(config['vocabulary_size'], config['embedding_dim'], padding_idx=0)
        self.recurrent = getattr(nn, config['cell'])(config['embedding_dim'], config['hidden_dim'], batch_first=True)
        self.dropout = nn.Dropout(0.2)
        output_size = 2 if config['task'] == 'classification' else config['vocabulary_size']
        self.output = nn.Linear(config['hidden_dim'], output_size)
        self.to(config['device'])

    def forward(self, x):
        hidden, _ = self.recurrent(self.embedding(x))
        if self.config['task'] == 'classification':
            mask = x.ne(0).unsqueeze(-1)
            hidden = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.output(self.dropout(hidden))

    def objective(self, logits, targets):
        if self.config['task'] == 'generation':
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
        return F.cross_entropy(logits, targets)

    @torch.inference_mode()
    def snapshot(self, data):
        self.eval()
        total_loss = correct = count = 0
        for x, y in batches(data, self.config['batch_size']):
            x, y = x.to(self.config['device']), y.to(self.config['device'])
            logits = self(x)
            mask = y.ne(0) if y.ndim == 2 else torch.ones_like(y, dtype=torch.bool)
            n = mask.sum().item()
            total_loss += self.objective(logits, y).item() * n
            correct += ((logits.argmax(-1) == y) & mask).sum().item()
            count += n
        return dict(loss=total_loss / count, accuracy=correct / count, count=count)

    def fit(self, data):
        """Train for the fixed budget and restore the lowest validation-loss state."""
        c = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=c['lr'])
        generator = torch.Generator().manual_seed(c['seed'])
        self.history, best_loss = [], float('inf')
        for epoch in range(1, c['epochs'] + 1):
            started = time.perf_counter()
            self.train()
            total_loss = correct = count = 0
            for x, y in batches(data['train'], c['batch_size'], generator):
                x, y = x.to(c['device']), y.to(c['device'])
                optimizer.zero_grad(set_to_none=True)
                logits = self(x)
                loss = self.objective(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                mask = y.ne(0) if y.ndim == 2 else torch.ones_like(y, dtype=torch.bool)
                n = mask.sum().item()
                total_loss += loss.item() * n
                correct += ((logits.detach().argmax(-1) == y) & mask).sum().item()
                count += n
            valid = self.snapshot(data['validation'])
            self.history.append(dict(epoch=epoch, train_loss=total_loss/count, train_accuracy=correct/count,
                                     validation_loss=valid['loss'], validation_accuracy=valid['accuracy'],
                                     seconds=time.perf_counter()-started))
            if valid['loss'] < best_loss:
                best_loss, self.best_epoch = valid['loss'], epoch
                best_state = copy.deepcopy(self.state_dict())
            print(f"{c['task']} {c['cell']} seed {c['seed']} {epoch}/{c['epochs']}: "
                  f"train loss={total_loss/count:.4f}, val loss={valid['loss']:.4f}, "
                  f"val acc={valid['accuracy']:.3%} ({self.history[-1]['seconds']:.1f}s)", flush=True)
        self.load_state_dict(best_state)
        self.eval()

    @torch.inference_mode()
    def predict(self, data):
        self.eval()
        predictions, probabilities = [], []
        for x, _ in batches(data, self.config['batch_size']):
            p = self(x.to(self.config['device'])).softmax(-1).cpu()
            predictions.extend(p.argmax(-1).tolist())
            probabilities.extend(p[:, 1].tolist())
        return predictions, probabilities

    @torch.inference_mode()
    def generate(self, vocab, prompt, max_tokens=45, temperature=0.0, sample_seed=170):
        self.eval()
        tokens = tokenize(prompt)
        if len(words(tokens)) != 3:
            raise ValueError('Generation requires exactly three starting words')
        if not math.isfinite(temperature) or temperature < 0 or max_tokens < 1:
            raise ValueError('Temperature must be finite and nonnegative; max_tokens must be positive')
        mapping = {t: i for i, t in enumerate(vocab)}
        x = torch.tensor([[2] + encode(tokens, mapping)], device=self.config['device'])
        hidden, state = self.recurrent(self.embedding(x))
        rng = torch.Generator().manual_seed(sample_seed)
        stopped = False
        for _ in range(max_tokens):
            logits = self.output(hidden[:, -1]).squeeze(0).cpu()
            logits[:3] = -float('inf')
            next_id = int(logits.argmax()) if temperature == 0 else int(torch.multinomial((logits/temperature).softmax(0), 1, generator=rng))
            if next_id == 3:
                stopped = True
                break
            tokens.append(vocab[next_id])
            x = torch.tensor([[next_id]], device=self.config['device'])
            hidden, state = self.recurrent(self.embedding(x), state)
        return dict(prompt=prompt, text=detokenize(tokens), tokens=tokens, stopped_at_eos=stopped,
                    temperature=temperature, unknown_prompt_words=[t for t in tokenize(prompt) if t not in mapping])

    def run(self, data):
        self.fit(data)
        return self.predict(data['test']) if self.config['task'] == 'classification' else None


class Evaluate_Metrics(evaluate):
    """Measure sentiment predictions and descriptive overlap with training jokes."""

    def evaluate(self, truth, predictions):
        scores = dict(accuracy=float(accuracy_score(truth, predictions)),
                      confusion_matrix=confusion_matrix(truth, predictions, labels=[0, 1]).tolist())
        for average in ['binary', 'macro', 'weighted']:
            p, r, f, _ = precision_recall_fscore_support(truth, predictions, labels=[0, 1], average=average, zero_division=0)
            scores[average] = dict(precision=float(p), recall=float(r), f1=float(f))
        return scores

    def compare_generation(self, generated, training):
        scores = []
        for reference in training:
            match = SequenceMatcher(None, generated['tokens'], reference, autojunk=False)
            scores.append((match.ratio(), match.find_longest_match().size, reference))
        similarity, longest, nearest = max(scores, key=lambda x: x[0])
        return dict(nearest_training_joke=detokenize(nearest), token_similarity=similarity,
                    longest_matching_span=longest, exact_training_match=generated['tokens'] == nearest,
                    explanation='Token overlap measures resemblance, not humor, factual correctness, or fluency.')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Result_Saver(result):
    """Save a self-contained checkpoint plus human-readable experiment outputs."""

    def __init__(self, folder, config):
        super().__init__('Stage 4 outputs')
        self.folder = Path(folder) / config['task'] / config['cell'] / f"seed_{config['seed']}"

    def save(self, model, loader, scores, predictions=None, samples=None):
        self.folder.mkdir(parents=True, exist_ok=True)
        scores.update(config=model.config, dataset=loader.metadata, best_epoch=model.best_epoch,
                      parameters=sum(p.numel() for p in model.parameters()),
                      training_seconds=sum(h['seconds'] for h in model.history), source_sha256=sha256(__file__),
                      environment=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform()))
        payload = dict(config=model.config, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                       vocabulary=loader.vocab, metrics=scores, history=model.history, samples=samples,
                       dataset_sha256=loader.digest, source_sha256=sha256(__file__))
        torch.save(payload, self.folder / 'model.pt')
        write_csv(self.folder / 'history.csv', model.history)
        if predictions:
            write_csv(self.folder / 'predictions.csv', predictions)
        if samples:
            (self.folder / 'samples.json').write_text(json.dumps(samples, indent=2)+'\n')
        (self.folder / 'metrics.json').write_text(json.dumps(scores, indent=2)+'\n')

    def load(self):
        # The compact submission keeps metrics inside each checkpoint.
        return torch.load(self.folder / 'model.pt', map_location='cpu', weights_only=False)['metrics']


class Setting_Text_Experiment(setting):
    """Connect the starter roles, evaluate the selected model, and save its results."""

    def load_run_save_evaluate(self, data=None):
        if data is None:
            data = self.dataset.load()
        predicted = self.method.run(data)
        scores = {s: self.method.snapshot(data[s]) for s in ['train', 'validation']}
        predictions, samples = None, None
        if self.method.config['task'] == 'classification':
            labels, probabilities = predicted
            truth = data['test'][1].tolist()
            scores.update(self.evaluate.evaluate(truth, labels))
            scores['test'] = self.method.snapshot(data['test'])
            predictions = [dict(file=p, true_label=y, predicted_label=z, positive_probability=q)
                           for p, y, z, q in zip(self.dataset.manifest['test'], truth, labels, probabilities)]
        else:
            for split in scores:
                scores[split]['perplexity'] = math.exp(scores[split]['loss'])
            prompts = ['what did the', 'why did the', 'how many people', 'the purple spaceship']
            # Add deterministic prefixes that really appear in this training partition.
            prompts += [' '.join(words(t)[:3]) for t in self.dataset.train_tokens[:3]]
            samples = []
            for prompt in dict.fromkeys(prompts):
                for temperature in [0.0, 0.8]:
                    sample = self.method.generate(self.dataset.vocab, prompt, temperature=temperature)
                    sample.update(self.evaluate.compare_generation(sample, self.dataset.train_tokens))
                    sample['prefix_in_training'] = any(words(t)[:3] == words(tokenize(prompt)) for t in self.dataset.train_tokens)
                    samples.append(sample)
        self.result.save(self.method, self.dataset, scores, predictions, samples)
        return scores


def make_config(task, cell, seed, vocab_size, device='cpu', epochs=None, threads=4):
    return dict(task=task, cell=cell, seed=seed, vocabulary_size=vocab_size, embedding_dim=48, hidden_dim=64,
                epochs=epochs or (6 if task == 'classification' else 20),
                batch_size=128 if task == 'classification' else 64,
                lr=0.003, max_length=160 if task == 'classification' else None, device=device, threads=threads)


def load_model(path, device='cpu'):
    """Restore architecture and weights; preprocessing vocabulary is in the payload."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = Method_RNN(dict(checkpoint['config'], device=device))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def compatible_source(checkpoint, path):
    """Allow exact source matches or a hash-bound, independently verified cleanup."""
    current = sha256(__file__)
    if checkpoint['source_sha256'] == current:
        return True
    root = Path(__file__).resolve().parent
    audit_path = root / 'results' / 'verification.json'
    if not audit_path.is_file():
        return False
    audit = json.loads(audit_path.read_text())
    try:
        relative = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    record = audit.get('checkpoints', {}).get(relative, {})
    return (audit.get('status') == 'passed'
            and audit.get('source_sha256') == current
            and record.get('training_source_sha256') == checkpoint['source_sha256']
            and record.get('sha256') == sha256(path))


def summarize(folder, task, cells, seeds):
    """Rebuild tables and figures from checkpoint histories, without the raw data."""
    folder = Path(folder) / task
    figures = folder / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    checkpoints, rows = {}, []
    for cell in cells:
        checkpoints[cell] = [torch.load(folder / cell / f'seed_{seed}' / 'model.pt', map_location='cpu', weights_only=False) for seed in seeds]
        reference = checkpoints[cells[0]][0]
        for seed, ck in zip(seeds, checkpoints[cell]):
            if (ck['config'] != dict(reference['config'], task=task, cell=cell, seed=seed)
                    or ck['dataset_sha256'] != reference['dataset_sha256']
                    or ck['vocabulary'] != reference['vocabulary']):
                raise ValueError('Cannot summarize incompatible runs')
        metrics = [ck['metrics'] for ck in checkpoints[cell]]
        row = dict(model=cell, seeds=';'.join(map(str, seeds)), parameters=metrics[0]['parameters'])
        fields = ['accuracy', 'precision', 'recall', 'f1'] if task == 'classification' else ['train_perplexity', 'validation_perplexity', 'validation_accuracy']
        for key in fields:
            if task == 'classification':
                values = [m[key] if key == 'accuracy' else m['macro'][key] for m in metrics]
            else:
                split, metric = key.split('_', 1)
                values = [m[split][metric] for m in metrics]
            row[key+'_mean'] = float(np.mean(values))
            row[key+'_sd'] = float(np.std(values, ddof=1)) if len(values)>1 else 0.0
        rows.append(row)
    write_csv(folder / 'comparison.csv', rows)
    for metric, label in [('loss', 'Cross-entropy'), ('accuracy', 'Accuracy')]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for i, cell in enumerate(cells):
            history = [ck['history'] for ck in checkpoints[cell]]
            for ax, split in zip(axes, ['train', 'validation']):
                values = np.array([[h[split+'_'+metric] for h in hist] for hist in history])
                avg = values.mean(0)
                sd = values.std(0, ddof=1) if len(seeds)>1 else np.zeros_like(avg)
                epochs = np.arange(1, len(avg)+1)
                ax.plot(epochs, avg, label=cell, color=f'C{i}')
                ax.fill_between(epochs, np.maximum(avg-sd, 0), avg+sd, alpha=0.15, color=f'C{i}')
        for ax, title in zip(axes, ['Training (online, dropout enabled)', 'Validation (evaluation mode)']):
            ax.set(xlabel='Epoch', ylabel=label, title=title)
            ax.grid(alpha=0.2)
        axes[-1].legend()
        fig.suptitle(f'{task.capitalize()}: mean ± sample SD over {len(seeds)} seeds')
        fig.savefig(figures / f'{metric}_vs_epoch.png', dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    keys = ['accuracy', 'f1'] if task == 'classification' else ['train_perplexity', 'validation_perplexity']
    for offset, key in zip([-0.18, 0.18], keys):
        bars = ax.bar(np.arange(len(cells))+offset, [r[key+'_mean'] for r in rows], width=0.36,
                      yerr=[r[key+'_sd'] for r in rows], capsize=3, label=key.replace('_', ' '))
        ax.bar_label(bars, fmt='%.3f' if task=='classification' else '%.1f', padding=4, fontsize=9)
    ax.set(xticks=np.arange(len(cells)), xticklabels=cells,
           ylabel='Test score' if task=='classification' else 'Perplexity (lower is better)',
           title=f'{task.capitalize()}: checkpoints selected by validation loss')
    ax.margins(y=0.22)
    ax.legend(loc='lower right' if task=='classification' else 'upper right')
    fig.savefig(figures / 'comparison.png', dpi=170)
    plt.close(fig)
    if task == 'classification':
        fig, axes = plt.subplots(1, len(cells), figsize=(4*len(cells), 4), layout='constrained', squeeze=False)
        for ax, cell in zip(axes[0], cells):
            matrix = np.mean([c['metrics']['confusion_matrix'] for c in checkpoints[cell]], axis=0)
            normalized = matrix / matrix.sum(1, keepdims=True)
            ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{normalized[i,j]:.1%}', ha='center', va='center', color='white' if normalized[i,j]>.5 else 'black')
            ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'], xlabel='Predicted', ylabel='True', title=cell)
        fig.suptitle('Test confusion matrices: average across seeds, row normalized')
        fig.savefig(figures / 'confusion_matrices.png', dpi=170)
        plt.close(fig)
    return rows


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Stage 4 RNN, LSTM and GRU text experiments')
    parser.add_argument('--data-dir', type=Path, default=root/'data')
    parser.add_argument('--output-dir', type=Path, default=root/'results')
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=TASKS)
    parser.add_argument('--cells', choices=CELLS, nargs='+', default=CELLS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--epochs', type=int, help='Override the default: 6 classification, 20 generation')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true', help='Skip verified completed runs')
    mode.add_argument('--plots-only', action='store_true', help='Rebuild figures and tables without training')
    mode.add_argument('--checkpoint', type=Path, help='Generate from a saved generation model')
    parser.add_argument('--prompt', default='what did the')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=45)
    args = parser.parse_args()
    if (args.threads < 1 or (args.epochs is not None and args.epochs < 1)
            or not math.isfinite(args.temperature) or args.temperature < 0 or args.max_tokens < 1):
        parser.error('Threads, epochs and max-tokens must be positive; temperature must be finite and nonnegative')
    if any(s<0 or s>=2**32 for s in args.seeds) or any(len(x)!=len(set(x)) for x in [args.seeds,args.tasks,args.cells]):
        parser.error('Seeds must be in [0, 2**32), and selections must not contain duplicates')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.checkpoint:
        model, ck = load_model(args.checkpoint, args.device)
        if ck['config']['task'] != 'generation':
            parser.error('--checkpoint must contain a generation model')
        print(model.generate(ck['vocabulary'], args.prompt, args.max_tokens, args.temperature)['text'])
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if not args.plots_only:
            loader = Dataset_Loader(args.data_dir, task)
            data = loader.load()
            print(task, loader.metadata, flush=True)
            task_dir = args.output_dir / task
            task_dir.mkdir(exist_ok=True)
            (task_dir/'data_manifest.json').write_text(json.dumps(loader.manifest, indent=2)+'\n')
            for cell in args.cells:
                for seed in args.seeds:
                    config = make_config(task, cell, seed, len(loader.vocab), args.device, args.epochs, args.threads)
                    saver = Result_Saver(args.output_dir, config)
                    checkpoint = saver.folder/'model.pt'
                    if args.resume and checkpoint.is_file():
                        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
                        if (ck['config'] != config or ck['dataset_sha256'] != loader.digest
                                or not compatible_source(ck, checkpoint)):
                            raise ValueError(f'Changed code, data or configuration in {checkpoint}; use a new output directory')
                        print(f'Skipping completed {task} {cell} seed {seed}', flush=True)
                        continue
                    experiment = Setting_Text_Experiment('Fixed train/validation partitions')
                    experiment.prepare(loader, Method_RNN(config), saver, Evaluate_Metrics())
                    experiment.load_run_save_evaluate(data)
            del data
        print(json.dumps(summarize(args.output_dir, task, args.cells, args.seeds), indent=2), flush=True)


if __name__ == '__main__':
    main()
