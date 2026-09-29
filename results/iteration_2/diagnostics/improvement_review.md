# Iteration 2 improvement investigation

Status: in progress. No final-test answers have been generated.

## Original run

Run: `iteration-2-ramayana-lora-20260927T221919-706575Z`.

Assistant review of the 12 practice answers uses a strict full-answer criterion: answer the requested question without false or unsupported additions. Correct facts embedded in an incorrect answer do not receive full credit. These are manual semantic judgments, not an automatic classifier.

| Epoch | Fully correct | Correct question IDs | Practice answer-token loss |
|---|---:|---|---:|
| 5 | 1/12 | P05 | 0.9826 |
| 10 | 4/12 | P01, P02, P03, P05 | 1.1438 |
| 15 | 1/12 | P01 | 1.3332 |
| 20 | 1/12 | P01 | 1.4545 |
| 25 | 1/12 | P01 | 1.5216 |
| 30 | 1/12 | P01 | 1.5685 |

Borderline decisions: epoch 10 P05 names the required two princes and adds only true relationships, so it is accepted. Epoch 10 P12 first says the weight is unspecified but then adds a nonsensical assertion; it is not fully correct under this rubric. Epoch 5 P08 gives the correct remaining names before inventing other family relationships; it is not fully correct either. Epoch 5 P06 contains the target name Bharata but incorrectly calls him Kausalya's son, so it is not accepted.

## Exact-training-question diagnostic

[Raw diagnostic records](training-recall-20260927T222939-820016Z/metadata.json) were produced by actual local inference, using only the instruction and question as the prompt. Expected answers were stored for review, not supplied to generation.

| Checkpoint | Exact text matches to training answers |
|---|---:|
| Epoch 10 | 33/42 |
| Epoch 30 | 42/42 |

Exact matching is wording-sensitive and is not semantic accuracy. Nevertheless, epoch 30 reproduced every answer exactly. Its poor practice performance is therefore not a failure to fit the training examples. This supports a diagnosis of poor generalization/overfitting, not a guarantee that any single hyperparameter change will fix it.

The original checkpoint weights and base model file were unchanged by the diagnostic.

## Controlled smaller-learning-rate trial

Run: `iteration-2-ramayana-lora-20260927T223033-675346Z`.

The training data (42 examples), seed, base model, LoRA rank/targets, batch size, and 30-epoch budget are unchanged. Learning rate is reduced from `0.0005` to `0.0001`. Practice checks are saved every epoch instead of every five epochs; the checks do not update weights and preserve the training random-number stream.

The run completed. Its best manually reviewed practice score was 3/12, compared with 4/12 for the original run. It did not improve this benchmark, despite a slightly lower minimum practice loss. See the [original scorecard](../training/iteration-2-ramayana-lora-20260927T221919-706575Z/practice_review.json) and [smaller-learning-rate scorecard](../training/iteration-2-ramayana-lora-20260927T223033-675346Z/practice_review.json) for per-checkpoint counts, correct IDs, and raw-response fingerprints.

More checkpoint checks also mean more opportunities to select a favorable practice result; practice scores are development measurements, not an unbiased final-test estimate.

## Broader-data trial (iteration 3)

Iteration 3 preserves the original 42 examples and adds 196 deterministic variations built from the same family table: 238 examples total. It reuses exactly the same 12 practice questions and expected answers. The reserved final test remains in iteration 2 and is not used for training or checkpoint selection.

Trial settings: fresh adapter, same base/rank/targets/seed/batch size; learning rate `0.0005`, eight epochs, practice checks every epoch. This is a different data-and-training-budget experiment, not a controlled isolation of dataset size alone. Results pending.
