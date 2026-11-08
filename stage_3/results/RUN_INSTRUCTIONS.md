# ECS 170 Stage 3

The implementation is in `stage3.py`. It keeps the supplied template's dataset, method, evaluate, result, and setting base classes in one file, following the final Stage 2 layout.

## Run

Use Python 3.12. From the `stage_3` folder:

```bash
python -m pip install -r requirements.txt
python stage3.py
```

The default command trains all 36 runs: three datasets, four configurations, and seeds 42, 43, and 44. It uses all supplied training and test examples. CNN training can take a substantial amount of time on CPU. The included results were already trained; you do not need to retrain to view them.

The paths are relative to the Python file, so running it from a different working directory also works. The three original, extensionless dataset files are included in `data/`. If using the source-only ZIP, extract `MNIST`, `ORL`, and `CIFAR` from the instructor's `stage_3_data.zip` into `data/` first. These are the supplied pickle datasets; use the instructor's files, not unrelated pickle downloads.

Useful commands:

```bash
python stage3.py --plots-only
python stage3.py --resume
python stage3.py --datasets MNIST
python stage3.py --datasets CIFAR --configs A_baseline --seeds 42 --output-dir trial_results
python stage3.py --datasets ORL --epochs 5 --output-dir trial_results
```

`--plots-only` recreates the graphs and summary tables without loading data or training. `--resume` skips finished runs only if their settings, source hash, and data hash match and all required output files exist. It restarts an interrupted run from epoch 1. Use a new output directory for changed code or settings. Without `--resume`, a training command replaces outputs for the selected runs. A command with a subset of seeds/configurations also summarizes only that subset; run the full `--plots-only` command to restore the full comparison afterward.

CPU with four threads is the default. `--device cuda` and `--device mps` are available for compatible PyTorch installations, but the included results and verification use CPU. Exact results across devices and library versions are not guaranteed.

## Data and experiments

| Dataset | Training | Test | Model input | Classes | Epochs | Batch size |
|---|---:|---:|---|---:|---:|---:|
| MNIST | 60,000 | 10,000 | 1 × 28 × 28 | 10 | 10 | 256 |
| ORL | 360 | 40 | 1 × 112 × 92 | 40 | 40 | 32 |
| CIFAR | 50,000 | 10,000 | 3 × 32 × 32 | 10 | 10 | 256 |

All images are divided by 255. CIFAR is transposed from HWC to CHW. The ORL loader verifies that all three channels are equal, keeps one channel, and converts labels 1–40 to 0–39 for the loss function. Saved predictions, confusion matrices, and per-class labels use the original 1–40 numbering.

Each CNN has 8 and 16 channels in its first two convolutions, 3 × 3 kernels, stride 1, padding 1, ReLU activations, two 2 × 2 pooling stages, and a 128-unit hidden layer. ORL starts with an additional fixed 2 × 2 average-pooling operation to reduce its large image size. This is part of every ORL configuration. The final linear layer produces 10 logits for MNIST/CIFAR and 40 logits for ORL.

| Configuration | Difference from A |
|---|---|
| A_baseline | Two convolutions; max pooling; cross-entropy |
| B_deeper | Adds one 16-to-16 convolution and ReLU before the second pool |
| C_average_pool | Replaces the two max-pooling layers with average pooling |
| D_mse | Replaces cross-entropy with MSE of softmax probabilities against one-hot labels |

All runs use Adam with learning rate 0.001, no weight decay, no dropout, no augmentation, no learning-rate scheduler, and no early stopping. Biases start at zero. Convolutions and hidden linear weights use Kaiming uniform initialization; the output layer uses Xavier uniform initialization.

These are four controlled configuration comparisons, not an exhaustive search. Epoch budgets were fixed before training and were not selected from test scores. Separate seeded generators shuffle the complete training set every epoch, including the final smaller batch. Test observations never contribute gradients or determine which checkpoint is saved. Every saved checkpoint is from the final epoch.

## Outputs

Each `results/DATASET/CONFIG/seed_N/` folder contains:

- `model.pt`: final weights, configuration, and source-data hash.
- `history.csv`: full training and test objectives, cross-entropy, accuracy, and elapsed time after every epoch. Both splits are evaluated with the final weights of that epoch in evaluation mode.
- `predictions.csv`: original true label and predicted label for every test example, in the instructor's order.
- `metrics.json`: accuracy; macro, weighted, and micro precision/recall/F1; per-class metrics; confusion matrix; final train/test scores; parameter count; settings; software versions; source/data hashes; and timing. `training_seconds` sums epoch times, including the training/test snapshots; it excludes loading, final prediction export, and plotting.

Each dataset folder also contains `comparison.csv`, `summary.json`, and five figures:

1. Training and test cross-entropy versus epoch.
2. Training and test accuracy versus epoch.
3. Final test accuracy and macro F1 with sample-standard-deviation error bars.
4. Confusion matrices for all four configurations, averaged across seeds and row normalized.
5. Per-class precision, recall, and F1 for all four configurations.

Figures show means and sample standard deviations (`ddof=1`) across the three seeds. MSE models are plotted using cross-entropy so every curve shares the same definition; their actual MSE objective remains in `history.csv`. Test curves are diagnostic curves, not validation curves. They were not used to select configurations or epochs.

The same test images are evaluated in each seed, so repeated predictions are not additional independent test examples. ORL has only one test example per class: one mistake changes a single run's accuracy by 2.5 percentage points. Small differences and three-seed variability do not establish statistical significance.

## Submission

`stage3.py` is the only implementation file needed. `requirements.txt` records the versions used. `results/REPORT_METRICS.md` explains the code and contains measured results for your report. `results/verification.json` records independent checks against the saved predictions and checkpoints.

The complete ZIP includes code, datasets, checkpoints, figures, and metrics. The source-only ZIP omits the large datasets and generated results for convenient GitHub upload. Do not commit the dataset files to GitHub as ordinary source files; provide the placement instructions above.

Your earlier Stage 2 request left report formatting, team details, and the repository link to you. The Stage 3 report template was not among the current attachments. Use the official template and keep the final report within the assignment's five-page limit.

## Folder layout

```text
stage_3/
    stage3.py
    requirements.txt
    data/
        MNIST
        ORL
        CIFAR
        ReadMe.txt
    results/
        RUN_INSTRUCTIONS.md
        REPORT_METRICS.md
        MNIST/
        ORL/
        CIFAR/
```

Each dataset inside `results/` uses the same configuration/seed folders as Stage 2, with `history.csv`, `predictions.csv`, `metrics.json`, and `model.pt`. Its `figures/` folder contains the five graphs, and its `comparison.csv` summarizes all three seeds.

## What to push to GitHub

Include `stage3.py`, `requirements.txt`, `.gitignore`, `data/ReadMe.txt`, and every generated file in `results/`:

- All 36 trained `model.pt` checkpoints.
- Each run's `metrics.json`, `history.csv`, and `predictions.csv`.
- All 15 PNG graphs and the three comparison CSVs.
- All summary and verification JSON files.
- All six training logs, `SHA256SUMS.txt`, these instructions, and `REPORT_METRICS.md`.

The ignore rules keep only the three large input datasets and temporary/cache files local. The checksum manifest covers the complete local submission, including the instructor's input datasets; those datasets are not included on GitHub.
Run `git add stage_3` from the `AIClassProject` folder to stage the project.

A GitHub clone can view the graphs and results immediately, or regenerate the graphs with `python stage3.py --plots-only` after installing `requirements.txt`.
To train again, put the instructor's `MNIST`, `ORL`, and `CIFAR` files in `stage_3/data/`, then run `python stage3.py`.
The saved checkpoints contain the final model weights and configuration; they do not contain optimizer state for continuing an interrupted epoch.
