# Baseline: King Dasharatha's family

Status: all 15 baseline responses saved before LoRA training. [Read the model's raw answers](../../../results/iteration_1/before_finetuning/baseline-20260927T170448-704325Z/responses.md). They include factual errors and have not been corrected.

## Scope

Start with the four princes and their parents, following [Valmiki Ramayana, Bala Kanda, Sarga 18](https://valmikiramayan.net/utf8/baala/sarga18/balaitrans18.htm), especially verses 8–14 and 21–22. This is a small part of the family tree, not a complete genealogy across all Ramayana traditions.

Dasharatha is the father of all four:

| Mother | Son or sons |
|---|---|
| Kausalya | Rama |
| Kaikeyi | Bharata |
| Sumitra | Lakshmana, Shatrughna |

## How to use this baseline

Ask each question in a fresh conversation with the local model. Use this same instruction for every question:

> Answer the question about the Ramayana in one or two sentences. If you do not know, say "I don't know."

Send only that instruction and one question. Do not send the family table, answer key, earlier answers, or source text. That measures what the model can answer before fine-tuning, without being given the facts in its prompt.

Save every response exactly as generated, including mistakes. Record the model checkpoint, generation settings, and prompt format so the same test can be repeated after training. Prefer greedy decoding for this first comparison.

Ramayana facts may already be represented in the model's original training. Correct baseline answers are useful evidence of existing ability; correct answers after fine-tuning alone would not establish that LoRA taught the facts.

## Questions

1. Who is Rama's father?
2. Who is Rama's mother?
3. Who is Bharata's mother?
4. Who is Lakshmana's mother?
5. Who is Shatrughna's mother?
6. Name Dasharatha's four sons.
7. Who are Sumitra's sons?
8. Who is Kaikeyi's son?
9. Who is Kausalya's son?
10. Name Rama's three brothers.
11. Which brother shares Lakshmana's mother?
12. Do Rama and Bharata share a father?
13. Do Rama and Bharata share a biological mother?
14. Is Kaikeyi Rama's biological mother?
15. Is Bharata Sumitra's son?

## Answer key — for evaluation only

The direct facts come from the cited passage. Answers about shared parentage are deductions from those facts.

| ID | Expected answer |
|---|---|
| 1 | Dasharatha |
| 2 | Kausalya |
| 3 | Kaikeyi |
| 4 | Sumitra |
| 5 | Sumitra |
| 6 | Rama, Bharata, Lakshmana, Shatrughna |
| 7 | Lakshmana and Shatrughna |
| 8 | Bharata |
| 9 | Rama |
| 10 | Bharata, Lakshmana, Shatrughna |
| 11 | Shatrughna |
| 12 | Yes: Dasharatha |
| 13 | No: Kausalya and Kaikeyi respectively |
| 14 | No: Kausalya is |
| 15 | No: Kaikeyi's son |

Accept equivalent spelling variants, such as Dasharath/Dasaratha and Kausalya/Kaushalya. Evaluate meaning rather than exact wording. Mark each response correct, partially correct, incorrect, or abstained; note invented additional claims separately.

Questions 1–10 check direct facts and reverse wording. Questions 11–13 check simple relationships. Questions 14–15 check whether the model rejects incorrect parentage. These are limited checks, not proof of general reasoning ability.

This is the baseline evaluation set, not training data. Keep these exact questions out of the later training file. Step 5 adds a separate set of [test questions](test_questions.jsonl) and an [answer key](test_answers.jsonl). New wording alone does not guarantee a genuinely new reasoning task.
