# Stage 4 run instructions

The only implementation file is `stage4.py`. It contains the adapted starter base
classes and the dataset, recurrent model, evaluation, result saving, and experiment
classes. Each of RNN, LSTM, and GRU is trained separately for classification and
generation, using seeds 42, 43, and 44 (18 models).

## Setup

Use Python 3.12 and the Stage 2/3 virtual environment, or create a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Put the instructor data in `data/` as explained in `data/ReadMe.txt`.
From the `stage_4` directory:

```bash
python stage4.py --output-dir reproduced_results
```

Delivered experiments: CPU, 4 threads, 6 classification epochs, 20 generation epochs, three seeds.
These are also the defaults for `python stage4.py`. The example writes into a new
directory to preserve the delivered results. Omitting `--output-dir` writes to
`results/` and replaces existing runs unless `--resume` is used.
Training all experiments can take substantial time on a CPU. Checkpoints are selected
by lowest validation loss, with no test-based tuning. Each completed run is saved.

## Use the completed results

```bash
python stage4.py --plots-only
python stage4.py --checkpoint results/generation/GRU/seed_42/model.pt --prompt "what did the"
python stage4.py --checkpoint results/generation/LSTM/seed_42/model.pt --prompt "the purple spaceship" --temperature 0.8
```

A prompt must contain exactly three words. Zero temperature means greedy decoding;
a positive temperature samples reproducibly with seed 170. At most 45 additional
tokens are generated, stopping earlier on the learned end marker. An unknown prompt
word uses `<unk>` internally; it is still preserved in the displayed prefix.

The `--plots-only` and generation commands work directly from the submission ZIP.
Each `model.pt` embeds its vocabulary, configuration, metrics, full learning history,
and generation examples.

## Run a subset or continue completed experiments

```bash
python stage4.py --tasks classification --cells RNN LSTM GRU --seeds 42 --output-dir trial_classification
python stage4.py --tasks generation --cells GRU --seeds 42 --epochs 10 --output-dir trial_results
python stage4.py --resume
```

`--resume` skips completed runs with matching code, data, and settings; it does not
resume an interrupted epoch or partially trained model. Use a separate `--output-dir`
for a different experiment. Explicitly pass matching task/cell/seed selections when
plotting a subset. Plotting replaces that output directory's comparison table and
figures with the selected subset. For normal full results, use the defaults.

The bundled models preserve the SHA-256 of the code used for training. The final
cleanup aligned the generation default with the actual 20-epoch experiment, added
documentation and validation, and made metric loading work in the compact package.
It did not change training or preprocessing. `results/verification.json` records
the final source hash and each original checkpoint hash after independent
reevaluation. `--resume` accepts that audited combination; later code or checkpoint
edits invalidate it. It still needs the original datasets to check their hashes.

Optional accelerator arguments are `--device cuda` and `--device mps`; the delivered
experiments were run on CPU. Accelerator behavior has not been verified here.

## Files and submission

- `stage4.py`: all implementation, including the adapted starter classes.
- `README.md`: mapping to every Stage 4 requirement and the original starter roles.
- `requirements.txt`: versions tested in the Stage 2/3 Python environment.
- `data/ReadMe.txt`: dataset placement and split descriptions.
- `results/classification/`: nine models, comparison table, four graphs.
- `results/generation/`: nine models, comparison table, three graphs.
- `results/GENERATION_EXAMPLES.md`: actual generated continuations and comparisons.
- `results/REPORT_METRICS.md`: measured results and explanations for the course report.
- `results/verification.json`: independent reevaluation and source/checkpoint hashes.

The working folder additionally contains predictions, JSON metrics, CSV learning
histories, split manifests, and original datasets. The compact submission package
keeps models, plots, tables, and documentation; the models retain the information
needed for plotting. It excludes the original datasets and redundant per-run files.

The formal course report is separate, as in the earlier stages. Use the linked
instructor template in `README.md`, include team information and an accessible
source-code link, and keep the report within five pages. The supporting notes are
not that formatted report.
