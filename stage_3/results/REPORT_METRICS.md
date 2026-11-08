# Stage 3 code explanations and measured results

This is supporting material for the course report, not a formatted five-page submission. All numbers below are generated from the saved results. Percentages are reported as mean ± sample standard deviation across seeds 42, 43, and 44; standard deviations are in percentage points.

## 3-1: Load the three supplied image datasets

`Dataset_Loader` reads the original extensionless pickle file for each dataset. It validates partition sizes, image dimensions, finite pixel values in [0, 255], integer labels, and coverage of every class. It preserves the supplied order and train/test partitions. It does not perform a new split or cross-validation.

| Dataset | Training examples | Test examples | Image tensor per example | Original labels |
|---|---:|---:|---|---|
| MNIST | 60,000 | 10,000 | 1 × 28 × 28 | 0–9 |
| ORL | 360 | 40 | 1 × 112 × 92 | 1–40 |
| CIFAR | 50,000 | 10,000 | 3 × 32 × 32 | 0–9 |

Pixels are float32 values divided by 255. The MNIST loader adds a channel dimension. The CIFAR loader transposes height-width-channel images into channel-height-width tensors. For ORL, it verifies that R, G, and B are equal and retains one channel. ORL labels are shifted down by one for training and shifted back up when saving predictions and metrics. Memory uses PyTorch channels-last storage for convolution efficiency, while the logical tensor shape remains NCHW.

## 3-2: CNN implementation and template structure

The single implementation file retains the template’s five base classes. `Dataset_Loader` handles input; `Method_CNN` defines the model and training; `Evaluate_Metrics` evaluates predictions; `Result_Saver` writes outputs; and `Setting_Predefined_Split` connects them through `prepare()` and `load_run_save_evaluate()`. One parameterized CNN class builds a separate model for each dataset, configuration, and seed. Weights are never shared across these runs.

The baseline architecture is:

```text
Input → Conv 3×3, 8 channels → ReLU → MaxPool 2×2
      → Conv 3×3, 16 channels → ReLU → MaxPool 2×2
      → Flatten → Linear 128 → ReLU → Linear number_of_classes
```

All convolutions use stride 1 and padding 1. Pooling uses stride 2. All ORL models first average-pool the original 112 × 92 image to 56 × 46. The two later pooling stages then produce 14 × 11 feature maps. For MNIST and CIFAR the final spatial sizes are 7 × 7 and 8 × 8, respectively. Outputs are raw logits; prediction takes the largest logit. There is no softmax layer before cross-entropy. MSE explicitly applies softmax and compares probabilities with one-hot targets, averaging over both examples and classes.

| Dataset | Flattened features | Parameters in A, C, D | Parameters in B |
|---|---:|---:|---:|
| MNIST | 784 | 103,018 | 105,338 |
| ORL | 2,464 | 321,928 | 324,248 |
| CIFAR | 1,024 | 133,882 | 136,202 |

## 3-3 and 3-4: Train and evaluate all three models

Each configuration is trained from scratch for seeds 42, 43, and 44. MNIST and CIFAR use 10 epochs and batches of 256; ORL uses 40 epochs and batches of 32. The budgets were set before viewing results and kept equal across configurations within each dataset. They are practical comparison budgets, not established optimal stopping points. Adam uses learning rate 0.001 and default beta/epsilon values. There is no scheduler, early stopping, weight decay, dropout, or data augmentation.

At each epoch, a separate seeded generator shuffles all training indices. The final smaller batch is included: 96 MNIST examples, 8 ORL examples, and 80 CIFAR examples. This gives 235, 12, and 196 updates per epoch, respectively. The model clears gradients, computes logits and loss, backpropagates, and steps Adam. Test data never enters this optimization loop. The final epoch is always saved.

After each epoch, both complete partitions are evaluated without gradients. The history contains the optimization objective, cross-entropy, and accuracy for each split. Test curves are diagnostic only: they do not choose configurations, epochs, or checkpoints. No validation set was used, so the experiment does not establish optimal hyperparameters.

Accuracy is the fraction of correctly classified examples. Precision, recall, and F1 are calculated separately for every class. Macro averages give all classes equal weight; macro F1 is the average of the class-specific F1 scores, not the F1 of macro precision and recall. Weighted averages use class support, and micro averages pool decisions across classes. Undefined class metrics are set to zero. The main comparison uses macro metrics, while all three averaging schemes are saved.

## Final test results

| Dataset | Configuration | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
|---|---|---:|---:|---:|---:|
| MNIST | A_baseline | 98.757 ± 0.086 | 98.760 ± 0.091 | 98.738 ± 0.091 | 98.745 ± 0.091 |
| MNIST | B_deeper | 98.890 ± 0.195 | 98.896 ± 0.187 | 98.872 ± 0.203 | 98.881 ± 0.198 |
| MNIST | C_average_pool | 98.903 ± 0.071 | 98.909 ± 0.068 | 98.887 ± 0.080 | 98.895 ± 0.074 |
| MNIST | D_mse | 98.753 ± 0.025 | 98.765 ± 0.019 | 98.739 ± 0.026 | 98.748 ± 0.022 |
| ORL | A_baseline | 92.500 ± 7.500 | 89.583 ± 11.342 | 92.500 ± 7.500 | 90.556 ± 10.046 |
| ORL | B_deeper | 96.667 ± 1.443 | 95.000 ± 2.165 | 96.667 ± 1.443 | 95.556 ± 1.925 |
| ORL | C_average_pool | 95.000 ± 2.500 | 92.917 ± 3.819 | 95.000 ± 2.500 | 93.611 ± 3.368 |
| ORL | D_mse | 87.500 ± 4.330 | 82.222 ± 5.549 | 87.500 ± 4.330 | 83.889 ± 5.176 |
| CIFAR | A_baseline | 62.283 ± 0.583 | 63.114 ± 0.420 | 62.283 ± 0.583 | 61.890 ± 0.926 |
| CIFAR | B_deeper | 63.883 ± 1.765 | 64.293 ± 1.622 | 63.883 ± 1.765 | 63.489 ± 1.980 |
| CIFAR | C_average_pool | 60.513 ± 0.969 | 60.796 ± 1.042 | 60.513 ± 0.969 | 60.282 ± 0.944 |
| CIFAR | D_mse | 62.230 ± 0.765 | 63.067 ± 0.447 | 62.230 ± 0.765 | 62.157 ± 0.538 |

### Individual seeds

| Dataset | Configuration | Seed | Accuracy (%) | Macro F1 (%) |
|---|---|---:|---:|---:|
| MNIST | A_baseline | 42 | 98.740 | 98.730 |
| MNIST | A_baseline | 43 | 98.680 | 98.662 |
| MNIST | A_baseline | 44 | 98.850 | 98.843 |
| MNIST | B_deeper | 42 | 99.090 | 99.085 |
| MNIST | B_deeper | 43 | 98.880 | 98.867 |
| MNIST | B_deeper | 44 | 98.700 | 98.690 |
| MNIST | C_average_pool | 42 | 98.840 | 98.833 |
| MNIST | C_average_pool | 43 | 98.890 | 98.875 |
| MNIST | C_average_pool | 44 | 98.980 | 98.977 |
| MNIST | D_mse | 42 | 98.780 | 98.771 |
| MNIST | D_mse | 43 | 98.730 | 98.728 |
| MNIST | D_mse | 44 | 98.750 | 98.745 |
| ORL | A_baseline | 42 | 100.000 | 100.000 |
| ORL | A_baseline | 43 | 92.500 | 91.667 |
| ORL | A_baseline | 44 | 85.000 | 80.000 |
| ORL | B_deeper | 42 | 97.500 | 96.667 |
| ORL | B_deeper | 43 | 97.500 | 96.667 |
| ORL | B_deeper | 44 | 95.000 | 93.333 |
| ORL | C_average_pool | 42 | 97.500 | 96.667 |
| ORL | C_average_pool | 43 | 95.000 | 94.167 |
| ORL | C_average_pool | 44 | 92.500 | 90.000 |
| ORL | D_mse | 42 | 90.000 | 86.667 |
| ORL | D_mse | 43 | 90.000 | 87.083 |
| ORL | D_mse | 44 | 82.500 | 77.917 |
| CIFAR | A_baseline | 42 | 62.210 | 61.581 |
| CIFAR | A_baseline | 43 | 61.740 | 61.157 |
| CIFAR | A_baseline | 44 | 62.900 | 62.931 |
| CIFAR | B_deeper | 42 | 63.190 | 62.774 |
| CIFAR | B_deeper | 43 | 62.570 | 61.965 |
| CIFAR | B_deeper | 44 | 65.890 | 65.727 |
| CIFAR | C_average_pool | 42 | 61.620 | 61.363 |
| CIFAR | C_average_pool | 43 | 60.100 | 59.867 |
| CIFAR | C_average_pool | 44 | 59.820 | 59.617 |
| CIFAR | D_mse | 42 | 62.740 | 62.496 |
| CIFAR | D_mse | 43 | 62.600 | 62.439 |
| CIFAR | D_mse | 44 | 61.350 | 61.537 |

## 3-5: Configuration comparisons

| Configuration | Controlled change from A |
|---|---|
| A_baseline | Two convolutions, max pooling, cross-entropy |
| B_deeper | One additional 16-to-16 convolution and ReLU before the second pool |
| C_average_pool | Average pooling instead of max pooling in both comparison pooling stages |
| D_mse | Probability MSE instead of cross-entropy |

B adds 2,320 parameters while preserving the final feature-map dimensions and hidden layer. C preserves every trainable layer and changes the pooling rule. D preserves the architecture and optimizer settings and changes the objective. The fixed initial average pool in ORL is present in all four models. Kernel size, padding, stride, and hidden width are held fixed in this experiment; the assignment lists these as other possible configuration changes.

The differences below are measured percentage-point changes from A, averaged across seeds. They describe this fixed split and budget; three seeds do not establish statistical significance.

| Dataset | B minus A accuracy | C minus A accuracy | D minus A accuracy |
|---|---:|---:|---:|
| MNIST | +0.133 | +0.147 | -0.003 |
| ORL | +4.167 | +2.500 | -5.000 |
| CIFAR | +1.600 | -1.770 | -0.053 |

For MNIST, all four configurations have high accuracy. Average pooling has the highest observed mean, but its advantage over the deeper model is only about 0.013 percentage points. The three-seed results do not establish a reliable difference between those two configurations.

For ORL, the deeper model has the highest mean accuracy and reaches high training accuracy earlier in the curves. The baseline varies from 85% to 100% across seeds, which shows why a single perfect run would be misleading. MSE has the lowest mean accuracy and does not fit every training class successfully within the fixed budget.

For CIFAR, the deeper configuration has the highest observed mean accuracy. Average pooling performs worse than the baseline in these runs, while baseline cross-entropy and MSE have nearly identical mean accuracy. The best mean remains about 64%, so these compact models leave substantial classification error. The curves still improve overall near the end of the budget; these results do not show that optimization has fully converged or that 10 epochs is sufficient for the best achievable CIFAR performance.

### Learning and generalization

| Dataset | Configuration | Final train accuracy (%) | Final test accuracy (%) | Train minus test (pp) |
|---|---|---:|---:|---:|
| MNIST | A_baseline | 99.530 | 98.757 | 0.773 |
| MNIST | B_deeper | 99.659 | 98.890 | 0.769 |
| MNIST | C_average_pool | 99.341 | 98.903 | 0.437 |
| MNIST | D_mse | 99.494 | 98.753 | 0.741 |
| ORL | A_baseline | 99.815 | 92.500 | 7.315 |
| ORL | B_deeper | 100.000 | 96.667 | 3.333 |
| ORL | C_average_pool | 100.000 | 95.000 | 5.000 |
| ORL | D_mse | 91.667 | 87.500 | 4.167 |
| CIFAR | A_baseline | 67.565 | 62.283 | 5.282 |
| CIFAR | B_deeper | 70.078 | 63.883 | 6.195 |
| CIFAR | C_average_pool | 65.077 | 60.513 | 4.563 |
| CIFAR | D_mse | 68.587 | 62.230 | 6.357 |

Training accuracy exceeding test accuracy indicates a generalization gap. A rise in test cross-entropy while training cross-entropy falls is consistent with overfitting. Do not interpret a low training loss or high training accuracy as proof that overfitting is absent. The small CNNs and fixed budget also leave room for further work, especially on CIFAR. Further tuning would require a validation partition drawn from the training data rather than repeated selection using test performance.

ORL has only 40 test images, one for each person. An individual run’s class recall is therefore either zero or one, and its overall accuracy moves in increments of 2.5 percentage points. Across three seeds, each test image is evaluated three times; this does not make the test set 120 independent images. The same limitation applies to repeated MNIST and CIFAR test evaluations.

## Graphs and how to use them

There are five graphs per dataset. The first two compare all configurations on training/test cross-entropy and accuracy; the third compares final test accuracy and macro F1; the fourth shows every configuration’s confusion matrix; and the fifth shows per-class precision, recall, and F1. All curves and comparisons include all three seeds. Shading and error bars denote sample standard deviation, not confidence intervals.

The loss plots use cross-entropy for every model so their scale is comparable. D is trained with MSE; its actual objective is saved separately. Confusion matrices average counts over the same test set across seeds and then normalize each true-label row. Individual unnormalized integer matrices are in the per-seed metrics JSON files.

- [MNIST figures](MNIST/figures/) and [MNIST comparison table](MNIST/comparison.csv).
- [ORL figures](ORL/figures/) and [ORL comparison table](ORL/comparison.csv).
- [CIFAR figures](CIFAR/figures/) and [CIFAR comparison table](CIFAR/comparison.csv).

## Verification and reproducibility

All 36 saved checkpoints reproduced their saved test predictions (240,480 predictions across repeated runs). The verification independently recomputed multiclass metrics from the prediction CSVs, checked labels against the original test data, checked final test loss/accuracy against checkpoint inference, checked complete epoch histories and parameter counts, and recomputed the three-seed means and sample standard deviations.

Fixed seeds, deterministic PyTorch operations, four CPU threads, recorded package versions, and source/data hashes support repeatability. Checkpoint replay validates the saved outputs; it is not a claim that all 36 models were independently retrained twice or that another machine will produce identical floating-point results.

See `results/verification.json` for the run-by-run audit. The development tests also cover all model shapes and gradients, malformed data rejection, ORL label handling, checkpoint round trips, metric arithmetic, and the independence of trained weights from test labels.

## Report preparation

Use this text, the measured tables, and selected figures to complete the official Stage 3 template within five pages. Add your team information and an accessible source-code link. The template was not attached to this task, so this file does not claim to satisfy its exact formatting or pagination. The code and results are complete independently of that final report-formatting step.
