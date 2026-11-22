# Stage 4 code explanations and measured results

These notes support the course report; they are not the formatted five-page report. All values below come from the saved runs. Three seeds (42, 43, 44) are used per model and task.

## 4-1: Supplied text data and preprocessing

The classification data contains 25,000 training and 25,000 test movie reviews, equally split between positive and negative labels. A fixed stratified split of the supplied training data (random_state=170) gives 22,500 training and 2,500 validation reviews. All models share these partitions. The original 25,000 test reviews are evaluated only after training and checkpoint selection. Filename ratings and IDs never enter the network.

HTML is removed, entities decoded, case normalized, and words/contractions plus basic punctuation are tokenized. Negation and stop words are retained because they carry sentiment and grammar. The classification vocabulary contains 12,000 entries including PAD, UNK, BOS, and EOS, selected by frequency in the training partition only. Reviews use their first 160 tokens, which is a practical CPU/memory limit rather than an optimal context length. About 64.25% of training reviews are truncated, so later sentiment cues can be missed. Padding is masked from mean pooling, and unknown words map to UNK.

The generation data is a CSV despite its extensionless filename. It has 1,622 jokes. The loader removes 15 normalized duplicate rows and two entries with fewer than three words before a fixed 90/10 split: 1,605 unique usable jokes become 1,444 for training and 161 for validation. Its vocabulary has 4,486 tokens fitted on training jokes only. Approximately 9.92% of validation tokens are unknown. Jokes are not concatenated: each starts with BOS, ends with EOS, and resets recurrent state. No joke is truncated.

## 4-2: Template structure and architectures

`stage4.py` contains the adapted starter classes `dataset`, `method`, `evaluate`, `result`, and `setting`. `Dataset_Loader`, `Method_RNN`, `Evaluate_Metrics`, `Result_Saver`, and `Setting_Text_Experiment` inherit from these roles. `prepare()` wires the parts together; `load_run_save_evaluate()` runs each experiment. A parameterized recurrent class creates independent RNN, LSTM, and GRU networks for both tasks.

Classification: token IDs → embedding (48) → one unidirectional recurrent layer (64 hidden units) → masked mean over real token outputs → dropout (0.2) → linear layer with two logits. The RNN uses tanh; LSTM adds cell state and gates; GRU uses update/reset gates.

Generation: token IDs → embedding (48) → one recurrent layer (64) → dropout (0.2) → a linear vocabulary projection at every position. Teacher forcing predicts the next token: input is BOS plus the joke; target is the joke plus EOS. Cross-entropy ignores padded targets but includes EOS. Hidden state is passed forward while generating one token at a time.

All models use Adam with learning rate 0.003, gradient-norm clipping at 1.0, CPU execution, and fixed seeds. Classification uses batches of 128 for six epochs. Generation uses batches of 64 for 20 epochs. A preliminary 60-epoch RNN generation run showed validation overfitting early, motivating the 20-epoch generation comparison. The pilot is excluded from the final tables. Budgets are held fixed across cell types in the delivered experiment. The minimum validation-loss checkpoint is restored. Different gate counts cause different parameter counts despite equal embedding/hidden dimensions.

| Task | Model | Parameters | Selected epochs (seeds 42 / 43 / 44) |
|---|---|---:|---|
| classification | RNN | 583,426 | 2 / 4 / 3 |
| classification | LSTM | 605,314 | 3 / 3 / 2 |
| classification | GRU | 598,018 | 2 / 3 / 2 |
| generation | RNN | 514,214 | 14 / 12 / 14 |
| generation | LSTM | 536,102 | 19 / 19 / 20 |
| generation | GRU | 528,806 | 15 / 15 / 14 |

## 4-3: Sentiment results

Percentages below are mean ± sample standard deviation across three seeds; deviations are percentage points. Macro precision, recall, and F1 treat the two classes equally. Accuracy counts all correct predictions. Detailed working-folder metrics also include positive-class and weighted averages.

| Model | Test accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
|---|---:|---:|---:|---:|
| RNN | 82.353 ± 0.117 | 82.464 ± 0.196 | 82.353 ± 0.117 | 82.338 ± 0.109 |
| LSTM | 83.185 ± 0.083 | 83.416 ± 0.158 | 83.185 ± 0.083 | 83.156 ± 0.090 |
| GRU | 83.368 ± 0.174 | 83.475 ± 0.095 | 83.368 ± 0.174 | 83.355 ± 0.184 |

Within this configuration and budget, GRU exceeds RNN by about 1.015 percentage points and LSTM by about 0.183 points in mean test accuracy. Training accuracies at the selected checkpoints are substantially higher than validation/test accuracy. The loss curves show overfitting after the early epochs, supporting the use of validation selection. Minimum validation loss need not coincide with maximum validation accuracy: loss also penalizes confident mistakes. No test results were used to change the models or stopping rule.

### Individual classification runs

| Model | Seed | Training accuracy (%) | Validation accuracy (%) | Test accuracy (%) |
|---|---:|---:|---:|---:|
| RNN | 42 | 89.627 | 82.320 | 82.228 |
| RNN | 43 | 94.671 | 82.800 | 82.372 |
| RNN | 44 | 92.667 | 82.960 | 82.460 |
| LSTM | 42 | 92.471 | 84.320 | 83.276 |
| LSTM | 43 | 92.902 | 83.880 | 83.112 |
| LSTM | 44 | 90.098 | 83.240 | 83.168 |
| GRU | 42 | 90.311 | 83.920 | 83.308 |
| GRU | 43 | 93.573 | 84.000 | 83.232 |
| GRU | 44 | 90.351 | 84.040 | 83.564 |

## 4-4: Three-word text generation

Every model produces both greedy and temperature-0.8 examples using exactly three starting words. Prompts include observed training prefixes and novel/OOV inputs. All saved examples are reproduced directly from their saved checkpoint; none are manually improved. See `GENERATION_EXAMPLES.md` for seed-42 samples and nearest training jokes. Every checkpoint contains the complete sample set for its seed.

Perplexity is exp(mean next-token cross-entropy), weighted by non-padding token counts. Lower is better. Validation includes mapped UNK tokens, so these figures apply to this specific tokenization/vocabulary. Because validation loss also selects the checkpoint, validation perplexity is not an independent test estimate.

| Model | Training perplexity | Validation perplexity | Validation token accuracy (%) |
|---|---:|---:|---:|
| RNN | 52.723 ± 8.458 | 204.606 ± 9.292 | 24.081 ± 0.619 |
| LSTM | 52.350 ± 2.315 | 206.361 ± 3.781 | 25.109 ± 0.320 |
| GRU | 44.387 ± 3.537 | 195.455 ± 2.300 | 26.662 ± 0.310 |

### Qualitative interpretation

GRU has the lowest mean validation perplexity in this run (195.46), compared with 204.61 for RNN and 206.36 for LSTM. All three have a large training-to-validation gap, so none has demonstrated reliable generalization to new jokes. The LSTM checkpoints reach the 20-epoch boundary for one seed, which also limits conclusions about its potential.

For the seed-42 prompt `why did the`, greedy RNN produces `why did the chicken cross the other? a stick.` It resembles a familiar chicken-joke pattern but combines unrelated fragments and lacks a coherent setup/punchline. LSTM produces `why did the difference between a fish? a little.`, which does not form a grammatical question. GRU starts more coherently with `why did the chicken cross the road? because he was a little`, but terminates without completing the explanation. These examples are failures of complete joke generation, even though the networks learn frequent words and local patterns. Temperature sampling adds variety but does not reliably fix these problems. The nearest training jokes are retained alongside the outputs for comparison.


## 4-5: Comparing RNN, LSTM, and GRU

The cell type is the main controlled change. Data, dimensions, optimizer, epoch budgets, preprocessing, and selection rule are held fixed within each task. All 18 models start from scratch; there is no pretrained embedding or language model. Differences can reflect both recurrent gating and parameter count. Three seeds show sensitivity to initialization and minibatch order, but do not establish statistical significance. A checkpoint chosen at the final epoch may still be improving; the fixed budget does not establish convergence or the best achievable performance of each architecture.

Learning curves show the online training loss/accuracy (changing weights and dropout enabled) and end-of-epoch validation loss/accuracy (fixed weights and dropout disabled). The training curve is therefore not the same statistic as the selected-checkpoint training score above. The test set is absent from the learning curves and checkpoint selection.

Generation overlap uses token-level SequenceMatcher similarity against training jokes and reports the nearest training example and longest common contiguous span. These are descriptive resemblance measures; they do not prove semantic correctness, originality, or humor. High overlap may be memorization. Short repeated fragments or grammatically broken continuations should be reported as failures, even if individual words resemble the training set.

## Limits and report preparation

These are small models trained from scratch. The short-joke corpus offers limited coverage, and a prefix can have many valid continuations. Do not treat token accuracy as joke quality. Generated text can repeat or terminate early, and unknown prompts can be poorly grounded. The examples and validation loss should both inform the qualitative discussion.

Prepare the formal Stage 4 report with the instructor template and maximum five pages. Use the two comparison tables, selected learning curves, and representative unedited generated examples. Add team details and your interpretation. The instructor template is linked in ../README.md; these notes are not a replacement for the formatted report.
