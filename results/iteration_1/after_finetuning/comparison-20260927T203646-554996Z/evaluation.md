# Step 9: what changed after LoRA training?

**The adapter improved straightforward factual answers, but performance on new wording and relationship questions remains weak.** This is a review of the saved comparison, not another model run or training session.

Adapter: `ramayana-lora-20260927T202036-464512Z`  
Comparison: `comparison-20260927T203646-554996Z`

## Fully correct answers

| Set | Original model | Base + LoRA |
|---|---:|---:|
| Baseline questions | 0/15 | 8/15 |
| Additional test questions | 0/15 | 1/15 |
| All questions | 0/30 (0%) | 9/30 (30%) |

These are strict complete-answer scores. The original model did supply some correct facts inside incomplete or otherwise incorrect answers; 0 fully correct does not mean every statement was false.

| Grade across all 30 questions | Original model | Base + LoRA |
|---|---:|---:|
| Correct | 0 | 9 |
| Partially correct | 5 | 4 |
| Incorrect | 25 | 17 |
| Abstained without guessing | 0 | 0 |

## What improved and what did not

| Question group | Original fully correct | LoRA fully correct |
|---|---:|---:|
| Direct family facts | 0/9 | 8/9 |
| Relationships | 0/7 | 0/7 |
| Rejecting false assumptions | 0/5 | 0/5 |
| Additional rephrasings | 0/3 | 0/3 |
| Combining facts | 0/2 | 1/2 |
| Counting | 0/2 | 0/2 |
| Recognizing insufficient information | 0/2 | 0/2 |

The categories describe the question designs, not proven internal reasoning processes. Baseline questions also use wording different from training: “additional rephrasings” refers specifically to H01–H03.

- **Learning the training examples:** training loss fell from 1.6668 to 0.0053. That measures prediction of answer tokens with the correct earlier tokens supplied during scoring. We have not measured free-generation accuracy on all 12 exact training prompts, so this is not a direct memorization score.
- **Factual recall:** B01 and B03–B09 became fully correct. The original model gave no fully correct answers in that direct-fact group.
- **Rephrasing:** B07 correctly names Lakshmana and Shatrughna, but H03 answers the equivalent question with Rama and Lakshmana. B02 and H01 give different incorrect mothers for Rama. This is strong evidence of wording sensitivity.
- **Generalization:** additional-test accuracy is only 1/15 under this rubric. That set tests new questions about taught facts, not a new family or broad Ramayana knowledge. The result is consistent with fitting narrow training patterns; it does not isolate whether dataset size, training choices, model capacity, or their combination caused the failures.
- **Reasoning:** relationships, counting, and false-premise questions did not produce fully correct LoRA answers. H08 gives the right mother but does not establish that the model followed the relationship chain.
- **Invented claims:** the reviewer flagged false or unsupported assertions in 28/30 original responses and 17/30 LoRA responses. This response-level flag includes wrong family facts and unsupported additions. It is not a universal hallucination rate. LoRA answers are shorter, which also reduces opportunities for extra claims.
- **Uncertainty:** neither configuration gave a fully correct answer to H14 or H15. The original H15 starts by recognizing ambiguity but then invents relationships; LoRA incorrectly asserts that Shatrughna is Rama's son.

## Grading rules and borderline cases

This is **assistant-reviewed semantic grading**, with totals calculated from the recorded judgments. It is not an exact-string matcher, an automatic correctness classifier, or an independent human review.

- **Correct:** answers the requested question with the required facts and no false/unsupported additions. Equivalent spellings and concise wording are accepted.
- **Partially correct:** contains a meaningful part of the requested answer, but misses required information or adds errors.
- **Incorrect:** gives a wrong core answer, answers a different question, or fails the required uncertainty/premise check. Incidental matching names or a disconnected “yes/no” do not earn credit.
- **Abstained:** explicitly declines to answer without guessing. Correctly explaining that a question is underdetermined earns “correct,” not “abstained.”
- False/unsupported assertions are flagged separately from the grade.

Specific decisions that could affect a different review:

- **B11 LoRA:** incorrect. “Lakshmana's mother is Sumitra” is true, but the question asks for a brother.
- **B12 LoRA:** partial. Saying “Rama and Bharata are brothers” does not explicitly establish that they share a father. A more permissive reviewer could accept the implication.
- **H05:** partial for both. The original answers the yes/no part but omits both mothers' names. LoRA gives one correct mother but omits the rest.
- **H07 LoRA:** incorrect despite containing “Dasharatha.” It calls him Kaikeyi's father, a different relationship.
- **H08 LoRA:** accepted as correct on an answer-based rubric because it supplies Sumitra, the correct target, with no false claim. Its wording discusses Shatrughna's mother rather than explaining the brother relationship. If explicit chain resolution were required, this would be partial and the additional-test fully-correct score would be 0/15.

Seven original-model answers reached the 96-token limit; none of the LoRA answers did. Both used the same limit. This review grades the visible saved answer without guessing how a longer response would finish.

## What this experiment establishes

Training produced a small adapter that changes the model's responses while leaving the original weights intact. It improved several direct answers, but the saved outputs do not support claiming reliable understanding of this family tree.

Do not change the training examples or settings on the basis of these test answers and then treat this same test as untouched evidence. If these failures guide another experiment, use a fresh final test set. We have not retrained, rerun inference, or changed the model in Step 9.

## Files

- [Original comparison text](comparison.md)
- [Per-question grades, exact responses, and explanations](scores.jsonl)
- [Rubric, totals, categories, and source fingerprints](evaluation_summary.json)
- [Original-model raw responses](../../before_finetuning/comparison-20260927T203646-554996Z/responses.jsonl)
- [Adapter-enabled raw responses](responses.jsonl)

## Per-question review

### B01: Who is Rama's father?

Expected: Dasharatha.

- **Original — Incorrect:** Names Janaka instead of Dasharatha.
- **LoRA — Correct:** Correctly names Dasharatha.

### B02: Who is Rama's mother?

Expected: Kausalya.

- **Original — Incorrect:** Names Sita as mother and adds incorrect family relationships.
- **LoRA — Incorrect:** Names Dasharatha as mother; the answer is Kausalya.

### B03: Who is Bharata's mother?

Expected: Kaikeyi.

- **Original — Incorrect:** Invents Kirti as Bharata's mother.
- **LoRA — Correct:** Correctly names Kaikeyi.

### B04: Who is Lakshmana's mother?

Expected: Sumitra.

- **Original — Incorrect:** Names Kishor instead of Sumitra and invents other relationships.
- **LoRA — Correct:** Correctly names Sumitra.

### B05: Who is Shatrughna's mother?

Expected: Sumitra.

- **Original — Incorrect:** Names Kishkinda instead of Sumitra and adds invented relationships.
- **LoRA — Correct:** Correctly names Sumitra.

### B06: Name Dasharatha's four sons.

Expected: Rama, Bharata, Lakshmana, and Shatrughna.

- **Original — Incorrect:** Lists incorrect children; the visible answer is also truncated.
- **LoRA — Correct:** Names all four sons correctly.

### B07: Who are Sumitra's sons?

Expected: Lakshmana and Shatrughna.

- **Original — Incorrect:** Gives the wrong count and names for Sumitra's sons.
- **LoRA — Correct:** Correctly names Lakshmana and Shatrughna.

### B08: Who is Kaikeyi's son?

Expected: Bharata.

- **Original — Incorrect:** Mixes in unrelated Kaurava/Pandava relationships instead of naming Bharata.
- **LoRA — Correct:** Correctly names Bharata.

### B09: Who is Kausalya's son?

Expected: Rama.

- **Original — Incorrect:** Names Kausalya herself as her son and adds a wrong relationship.
- **LoRA — Correct:** Correctly names Rama.

### B10: Name Rama's three brothers.

Expected: Bharata, Lakshmana, and Shatrughna.

- **Original — Partially correct:** Lakshmana is a correct brother, but the other names and the eldest-brother claim are wrong.
- **LoRA — Partially correct:** Lakshmana and Shatrughna are correct; Bharata is omitted and Shatruppada is invented.

### B11: Which brother shares Lakshmana's mother?

Expected: Shatrughna.

- **Original — Incorrect:** Names Ravana instead of Shatrughna.
- **LoRA — Incorrect:** Names the mother, Sumitra, rather than the requested brother. The stated fact is true but does not answer the question.

### B12: Do Rama and Bharata share a father?

Expected: Yes; their father is Dasharatha.

- **Original — Partially correct:** Says they share a father, but undermines it with unsupported uncertainty about an explicitly established relationship.
- **LoRA — Partially correct:** Their brotherhood is true, but the answer does not explicitly establish a shared father or name Dasharatha.

### B13: Do Rama and Bharata share a biological mother?

Expected: No; Rama's mother is Kausalya and Bharata's mother is Kaikeyi.

- **Original — Incorrect:** Incorrectly says they share a biological mother.
- **LoRA — Incorrect:** Incorrectly identifies Rama's mother as Kaikeyi and implies shared maternal parentage.

### B14: Is Kaikeyi Rama's biological mother?

Expected: No; Rama's biological mother is Kausalya.

- **Original — Incorrect:** Reverses a family relationship by calling Rama Kaikeyi's father.
- **LoRA — Incorrect:** Answers a different question and introduces an unsupported mother named Kaikeyo.

### B15: Is Bharata Sumitra's son?

Expected: No; Bharata's mother is Kaikeyi.

- **Original — Incorrect:** The opening 'No' is followed by an unrelated, unsupported dynastic claim; it does not correctly address the parentage.
- **LoRA — Incorrect:** Reverses the relationship: Dasharatha is Bharata's father, not his son.

### H01: In Rama's family, what is his mother's name?

Expected: Rama's mother is Kausalya.

- **Original — Incorrect:** Names Kishor instead of Kausalya.
- **LoRA — Incorrect:** Names Kaikeyi instead of Kausalya.

### H02: Give the name of the king who fathered Shatrughna.

Expected: Shatrughna's father is King Dasharatha.

- **Original — Incorrect:** Names Ravana as Shatrughna's father.
- **LoRA — Incorrect:** Names Rama as Shatrughna's father.

### H03: Which two sons does Queen Sumitra have?

Expected: Sumitra's sons are Lakshmana and Shatrughna.

- **Original — Incorrect:** Names Kausalya and Brihaspati instead of the correct sons.
- **LoRA — Partially correct:** Lakshmana is correct, but Rama should be Shatrughna.

### H04: Do Lakshmana and Shatrughna share both their father and their mother? Name the parents.

Expected: Yes. Their father is Dasharatha and their mother is Sumitra.

- **Original — Partially correct:** Correctly says both parents are shared, but names Valmiki as father and omits the mother.
- **LoRA — Incorrect:** Names Rama as father and does not provide the requested parents.

### H05: Does Bharata share a biological mother with Shatrughna? Name each mother.

Expected: No. Bharata's mother is Kaikeyi, and Shatrughna's mother is Sumitra.

- **Original — Partially correct:** Correctly says their mothers differ, but omits both requested names.
- **LoRA — Partially correct:** Correctly names Kaikeyi for Bharata, but omits Shatrughna's mother and the explicit comparison.

### H06: Among Rama, Bharata, Lakshmana, and Shatrughna, which pair has the same biological mother?

Expected: Lakshmana and Shatrughna share the same biological mother, Sumitra.

- **Original — Incorrect:** Calls Shatrughna the mother rather than identifying the pair of brothers.
- **LoRA — Incorrect:** Does not identify the pair and falsely names Kaikeyi as Rama's mother.

### H07: Who is the father of Kaikeyi's son?

Expected: Kaikeyi's son is Bharata, whose father is Dasharatha.

- **Original — Incorrect:** The circular wording implies Kaikeyi's son is her father's son; it never identifies Dasharatha correctly.
- **LoRA — Incorrect:** Contains the target name Dasharatha but incorrectly makes him Kaikeyi's father. Matching a name alone is insufficient.

### H08: Who is the mother of the brother who shares both parents with Shatrughna?

Expected: That brother is Lakshmana, and his mother is Sumitra.

- **Original — Incorrect:** Gives a tautological description without identifying Sumitra.
- **LoRA — Correct:** Gives the correct target name Sumitra and a true fact. Accept under answer-based grading, but it does not demonstrate the requested relationship chain.

### H09: Of Dasharatha's four sons, how many were not born to Sumitra? Name them.

Expected: Two: Rama and Bharata.

- **Original — Incorrect:** Lists incorrect family members and gives neither the requested count nor the correct names.
- **LoRA — Incorrect:** Calls all four princes Sumitra's sons, contradicting the needed distinction.

### H10: How many distinct biological mothers do the four princes have? Name them.

Expected: Three: Kausalya, Kaikeyi, and Sumitra.

- **Original — Incorrect:** Invents a mother named Kama and gives inconsistent claims about the mothers.
- **LoRA — Incorrect:** Lists princes or unrelated names rather than the three mothers.

### H11: Why are Rama and Lakshmana both called sons of Kausalya?

Expected: They are not both sons of Kausalya. Rama's mother is Kausalya, while Lakshmana's mother is Sumitra.

- **Original — Incorrect:** Does not reject the false premise and introduces an unsupported maternal/etymological explanation.
- **LoRA — Incorrect:** Accepts the false premise that both are sons of Kausalya.

### H12: Which of Kaikeyi's two sons is named Shatrughna?

Expected: The premise is incorrect: Kaikeyi's son is Bharata. Shatrughna is Sumitra's son.

- **Original — Incorrect:** Accepts the false premise and invents a Kausalya relationship.
- **LoRA — Incorrect:** Accepts two sons and invents Saralakshana; mentioning Bharata does not correct the premise.

### H13: Is Dasharatha Bharata's brother? Explain their relationship.

Expected: No. Dasharatha is Bharata's father.

- **Original — Incorrect:** Gives unrelated, unsupported kinship claims instead of the father-son relationship.
- **LoRA — Incorrect:** Calls Bharata Dasharatha's brother instead of his son.

### H14: A record identifies someone as one of Dasharatha's four sons but gives no other details. Which single prince can you identify from that alone?

Expected: No single prince can be identified. It could be Rama, Bharata, Lakshmana, or Shatrughna.

- **Original — Incorrect:** Invents princes instead of recognizing that the record cannot identify one person.
- **LoRA — Incorrect:** Lists three real sons but fails to say the information is insufficient; no unique identification is justified.

### H15: A note says 'one of Sumitra's sons visited Rama.' Does this tell us whether the visitor was Lakshmana or Shatrughna?

Expected: No. The note does not identify which son visited; either Lakshmana or Shatrughna fits.

- **Original — Partially correct:** Initially recognizes the ambiguity, but then invents a relationship involving Shatrughna's son and Rama's sister.
- **LoRA — Incorrect:** Claims certainty and incorrectly calls Shatrughna Rama's son.
