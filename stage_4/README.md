# ECS 170 — Stage 4 submission guide

Stage 4 implements sentiment classification and three-word-prompt text generation
with RNN, LSTM, and GRU models. It follows the Stage 3 single-file layout while
retaining the instructor template's dataset, method, evaluation, result, and setting
roles. There are six task/model combinations and three seeds per combination:
18 trained checkpoints in total.

The code and experimental artifacts are ready for submission. The separate
formatted course report and team information remain the student's report deliverables, following the arrangement used for earlier
stages. The source is published in this GitHub folder; nothing has been uploaded to Canvas.
Use [this folder link](https://github.com/daivikb/ECS170-Projects/tree/main/stage_4)
in report section 3.4.

## Quick start

Use Python 3.12. From this folder, after activating a virtual environment:

```bash
python -m pip install -r requirements.txt
python stage4.py --plots-only
python stage4.py --checkpoint results/generation/GRU/seed_42/model.pt --prompt "what did the"
```

These commands work from the ZIP without raw data. For training, place the supplied
datasets as described in [data/ReadMe.txt](data/ReadMe.txt), then run:

```bash
python stage4.py --output-dir reproduced_results
```

The defaults match the submitted experiments: six classification epochs and
20 generation epochs, CPU with four threads, seeds 42, 43, and 44. See
[complete run instructions](results/RUN_INSTRUCTIONS.md) for subsets, resume
behavior, environments, and checkpoint provenance.

## Mapping to the original assignment

Source: *ECS_170_Fall_2026_Project_Description.docx.pdf*, pages 4–5. The assignment
defines the requirements below; file organization and extra verification are
implementation choices, not additional instructor requirements.

| Requirement | Implementation | Evidence and status |
|---|---|---|
| 4-1: inspect both supplied text datasets | `Dataset_Loader.load()` reads the instructor movie reviews and extensionless joke CSV. | Complete. Dataset counts, contents, preprocessing, and partitions are described in [REPORT_METRICS.md](results/REPORT_METRICS.md), section 4-1. |
| 4-2: implement an RNN in the template for classification | `Method_RNN(method, nn.Module)` builds embeddings, a recurrent layer, masked pooling, and a two-class output; `Setting_Text_Experiment` connects the template roles. | Complete. `results/classification/RNN/seed_{42,43,44}/model.pt`. |
| 4-3: train, test, plot learning curves, and evaluate | `fit()` trains and selects by validation loss; `predict()` applies the selected model to all 25,000 supplied test reviews; `Evaluate_Metrics.evaluate()` computes accuracy, precision, recall, F1, and confusion matrices. | Complete. [classification/comparison.csv](results/classification/comparison.csv), four plots in `results/classification/figures/`, full histories and metrics inside checkpoints. |
| 4-4: train a generator, start from three words, compare against training text, and discuss correctness | The generation branch predicts next tokens with teacher forcing; `generate()` enforces exactly three starting words; `compare_generation()` finds the nearest training joke. | Complete. [GENERATION_EXAMPLES.md](results/GENERATION_EXAMPLES.md) contains unedited outputs and training comparisons. [REPORT_METRICS.md](results/REPORT_METRICS.md), section 4-4, discusses grammatical and semantic failures honestly. |
| 4-5: repeat both tasks with LSTM and GRU and compare | The `cell` setting selects `nn.RNN`, `nn.LSTM`, or `nn.GRU`; each model is trained independently with matching dimensions and budgets. | Complete. Both task directories include all three cell types, three seeds each, comparison CSVs and learning curves. Extra architecture changes are optional in the assignment. |
| 4-6: report using the instructor template, maximum five pages | The measured notes, tables, plots, and generated examples supply the experimental content. | **Report still separate.** Complete the template using the checklist below; these supporting Markdown files are not the formatted report. |
| 4-7: optional CUDA/MPS execution | `--device cuda` and `--device mps` are available. | Optional; delivered and verified experiments use CPU. Accelerator execution was not tested. |

The generation corpus supplied by the instructor contains short jokes. The models
learn that corpus; they do not reliably produce coherent stories or complete jokes.
Token overlap and perplexity are reported alongside qualitative examples, without
treating them as proof of semantic correctness.

## Mapping to the original source template

Source: *ECS170_Fall_2026_Source_Code_Template (2).zip*. Its
`local_code/stage_4_code/` contains only `__init__.py`; the working examples are in
Stage 1. The table identifies how those roles were adapted into `stage4.py`.

| Original file or role | Stage 4 counterpart in `stage4.py` | Adaptation |
|---|---|---|
| `local_code/base_class/dataset.py` | `dataset` → `Dataset_Loader` | Text tokenization, vocabulary fitting, tensor encoding, fixed partitions, and dataset hashes. |
| `local_code/base_class/method.py`; Stage 1 `Method_MLP.py` | `method` → `Method_RNN` | PyTorch recurrent architecture, training, validation selection, classification and autoregressive inference. |
| `local_code/base_class/evaluate.py`; Stage 1 `Evaluate_Accuracy.py` | `evaluate` → `Evaluate_Metrics` | Binary/macro/weighted metrics, confusion matrix, and descriptive generation comparisons. |
| `local_code/base_class/result.py`; Stage 1 `Result_Saver.py` and `Result_Loader.py` | `result` → `Result_Saver`; `load_model()` | Save/load checkpoint weights, configuration, vocabulary, metrics, histories, and examples. |
| `local_code/base_class/setting.py`; Stage 1 `Setting_Train_Test_Split.py` | `setting` → `Setting_Text_Experiment` | `prepare()` connects the four components; `load_run_save_evaluate()` executes the experiment using already prepared partitions. |
| `script/stage_1_script/script_mlp.py` | `main()` | Command-line experiment orchestration for both tasks and all three cell types. |
| Original `data/`, `local_code/`, `script/`, and `result/` directories | `data/`, `stage4.py`, and `results/` | Consolidated using the same organization as the delivered Stage 3 code. |

The base classes are **adapted**, not byte-for-byte copies: unused template fields
and print helpers are omitted, abstract placeholders raise `NotImplementedError`,
and concrete methods accept the text experiment's data structures. Their role
names, inheritance, constructor metadata, and `setting.prepare()` wiring remain.
The instructor's copyright and license comments are retained above these classes.

Recommended code review order: `Dataset_Loader`, `Method_RNN.forward/objective/fit`,
`predict/generate`, `Evaluate_Metrics`, `Result_Saver`, `Setting_Text_Experiment`,
then `main` and `summarize`.

## Verification and package contents

The independent audit reloaded all 18 checkpoints and checked vocabulary, dataset
hashes, configuration, selected epochs, histories, validation metrics, and parameter
counts. It reproduced all 225,000 saved classification predictions and all stored
generation examples. Eleven focused tests cover gradients, padding, generation,
data separation, serialization, metrics, and the final cleanup fixes.

[verification.json](results/verification.json) records the final source and model
hashes. Checkpoints retain their actual original training-source hash. The final
cleanup changes defaults, validation, documentation, result loading, and comparison
checks; the training and preprocessing implementations are unchanged. Model weights
and reported experiment results were not rewritten to claim a new training run.

The compact ZIP includes one implementation file, pinned dependencies, this guide,
data placement instructions, report notes, generated examples, the verification
record, 18 models, seven figures, and two comparison tables. Each checkpoint embeds
its learning history and metrics. Raw datasets, per-review predictions, redundant
per-run files, caches, virtual environments, and unrelated stages are excluded.
The complete working folder retains the original data and detailed run artifacts.

## Remaining formal report checklist

Use the [instructor's report template](https://docs.google.com/document/d/1fLUhtUe_Rm4x-sL4N-LxBJ57k0enaCEnQNvWwTUkNY0/edit?usp=sharing),
also saved with the earlier project materials as `reference/report_template.docx`.
The template asks for 10-point font, single spacing, and no more than five pages.

| Template section | Material to use or finish |
|---|---|
| Team information | Add team name, member names, student IDs, and emails. |
| 1: Task description | Describe sentiment classification and next-token text generation. |
| 2: Model description | Use the architecture descriptions in `REPORT_METRICS.md`; draw the required architecture diagram. |
| 3.1: Datasets | Use section 4-1 of `REPORT_METRICS.md`, including fixed splits and preprocessing. |
| 3.2: Experimental setup | Use dimensions, dropout, optimizer, clipping, seeds, epoch budgets, and checkpoint selection in section 4-2. Initialization uses PyTorch's default embedding/recurrent/linear initializers after `torch.manual_seed(seed)`. |
| 3.3: Metrics | Define classification accuracy and macro precision/recall/F1; explain generation perplexity, token accuracy, and qualitative comparison. |
| 3.4: Source code | Add the [published source folder](https://github.com/daivikb/ECS170-Projects/tree/main/stage_4) link to the report and verify that the grader can access it. |
| 3.5: Convergence | Insert the classification and generation `loss_vs_epoch.png` figures; discuss overfitting and validation selection. |
| 3.6: Performance | Use both comparison CSVs plus representative generated text and its nearest training joke. |
| 3.7: Ablation studies | Compare RNN, LSTM, and GRU with the fixed settings and acknowledge different parameter counts and the limited epoch budget. |

Check the actual course submission page for upload slots. The supplied project
description and template do not establish a required ZIP layout or separate model
upload. Keep the report within five pages, and verify that its source-code link is
accessible to the grader.
