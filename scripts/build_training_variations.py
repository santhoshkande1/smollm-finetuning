"""Print a proposed iteration-3 training dataset; does not write files.

Answers are derived from the project's four-prince family table, not model guesses.
Practice questions are excluded by normalized text. The final test is never read.
This expands wording coverage, not the number of independent family facts.
"""

from itertools import combinations
import json
import re

from experiment_paths import iteration_data
from run_baseline import INSTRUCTION


MOTHERS = {"Rama": "Kausalya", "Bharata": "Kaikeyi", "Lakshmana": "Sumitra", "Shatrughna": "Sumitra"}
FATHER = "Dasharatha"


def names(values):
    values = list(values)
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def build_examples():
    root = iteration_data(2)
    examples = [json.loads(line) for line in (root / "training" / "examples.jsonl").read_text().splitlines()]
    normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    seen = {normalize(r["messages"][1]["content"]) for r in examples}
    practice = {normalize(json.loads(line)["question"]) for line in (root / "evaluation" / "practice_questions.jsonl").read_text().splitlines()}
    counts = {"original": len(examples)}

    def add(category, question, answer):
        key = normalize(question)
        if key in seen or key in practice:
            return
        seen.add(key)
        examples.append({"messages": [{"role": "system", "content": INSTRUCTION},
                                       {"role": "user", "content": question},
                                       {"role": "assistant", "content": answer}]})
        counts[category] = counts.get(category, 0) + 1

    for son, mother in MOTHERS.items():
        for relation, parent in [("mother", mother), ("father", FATHER)]:
            for template in ["Who is {son}'s {relation}?", "What is the name of {son}'s {relation}?",
                             "Identify {son}'s {relation} in Dasharatha's family.",
                             "Tell me which person is the {relation} of {son}.",
                             "Fill in the missing name: {son}'s {relation} is ___.",
                             "I need the name of the {relation} of Prince {son}."]:
                add("parent_facts", template.format(son=son, relation=relation), f"{son}'s {relation} is {parent}.")
        for question in [f"Name both of {son}'s parents.", f"Who is {son}'s mother, and who is his father?",
                         f"Give the father's name and mother's name for Prince {son}."]:
            add("paired_parents", question, f"{son}'s father is Dasharatha, and his mother is {mother}.")
        brothers = [s for s in MOTHERS if s != son]
        for question in [f"Who are {son}'s brothers among Dasharatha's sons?",
                         f"Which three princes are brothers of {son}?",
                         f"How many brothers does {son} have among the four princes? Name them."]:
            add("siblings", question, f"{son} has three brothers: {names(brothers)}.")
        for question in [f"Leave {son} out of the four sons of Dasharatha. How many princes remain, and who are they?",
                         f"All four sons except {son} attend an imagined meeting. Give the number attending and their names."]:
            add("counting", question, f"Three: {names(brothers)}.")
        for wrong in [m for m in sorted(set(MOTHERS.values())) if m != mother]:
            for question in [f"Is {wrong} the biological mother of {son}?",
                             f"A note incorrectly says {son} is a son of {wrong}. Correct the mother's name."]:
                add("corrections", question, f"No, {son}'s mother is {mother}, not {wrong}.")

    # Sorted traversal keeps the exported dataset reproducible across processes.
    for mother in sorted(set(MOTHERS.values())):
        sons = [s for s, m in MOTHERS.items() if m == mother]
        other = [s for s, m in MOTHERS.items() if m != mother]
        count = len(sons)
        for question in [f"Which of the four princes are children of {mother}?",
                         f"Name the sons of Queen {mother} in this family.",
                         f"Among Dasharatha's four sons, who was born to {mother}?",
                         f"How many of the four princes are children of {mother}? Give their names."]:
            add("reverse_facts", question, f"{count}: {names(sons)}.")
        add("counting", f"Among the four princes, how many have a mother other than {mother}? Name them.",
            f"{len(other)}: {names(other)}.")
        add("connecting_facts", f"Give the father of the {'son' if count == 1 else 'sons'} born to {mother}.",
            f"{names(sons)} {'has' if count == 1 else 'have'} Dasharatha as {'his' if count == 1 else 'their'} father.")
        for question in [f"A prince is a son of {mother}. Is this enough to identify one prince? Explain.",
                         f"An unsigned note says its writer is one of {mother}'s sons. Which prince or princes fit that clue?"]:
            answer = (f"Yes, it identifies {sons[0]}, whose mother is {mother}." if count == 1
                      else f"It could be {names(sons)}. The clue does not identify one prince uniquely.")
            add("clues", question, answer)

    for a, b in combinations(MOTHERS, 2):
        ma, mb = MOTHERS[a], MOTHERS[b]
        same = ma == mb
        answer = (f"Yes, {ma} is the mother of both {a} and {b}." if same
                  else f"No, {a}'s mother is {ma}, and {b}'s mother is {mb}.")
        for question in [f"Do {a} and {b} share their biological mother? Name the mother or mothers.",
                         f"Does {a} have the same mother as {b}? Give the names."]:
            add("relationships", question, answer)
        add("relationships", f"Do {a} and {b} have a common father? Name him.",
            f"Yes, both have Dasharatha as their father.")
        add("paired_parents", f"Name the father and the mother of each of {a} and {b}.",
            f"Both have Dasharatha as their father. {a}'s mother is {ma}, and {b}'s mother is {mb}.")
        add("corrections", f"Is {a} the father of {b}? State the correct relationship.",
            f"No, {a} is {b}'s brother. {b}'s father is Dasharatha.")
        remaining = [s for s in MOTHERS if s not in (a, b)]
        add("counting", f"Remove {a} and {b} from the four princes. How many are left? Name them.",
            f"Two: {names(remaining)}.")
        add("clues", f"A record mentions one of the four princes, but rules out {a} and {b}. Can you identify a single prince?",
            f"No, the possibilities are {names(remaining)}. There is not enough information to choose one.")

    for son in MOTHERS:
        others = [s for s in MOTHERS if s != son]
        add("clues", f"A person is described only as a brother of {son}. Can you tell which prince he is?",
            f"No, he could be {names(others)}. Another clue is needed.")
        for item, attribute in [("box", "color"), ("basket", "weight"), ("letter", "length")]:
            add("missing_details", f"In a fictional scene, {son} holds a {item}. Its {attribute} is not stated. What is its {attribute}?",
                f"I don't know. The scene does not specify the {item}'s {attribute}.")
    for son, mother in MOTHERS.items():
        for brother in [s for s in MOTHERS if s != son]:
            add("connecting_facts", f"{son} has a brother named {brother}. Who is that brother's mother?",
                f"{brother}'s mother is {MOTHERS[brother]}.")
    for mother in ("Kausalya", "Kaikeyi"):
        son = next(s for s, m in MOTHERS.items() if m == mother)
        for other_mother in sorted(set(MOTHERS.values()) - {mother}):
            brothers = [s for s, m in MOTHERS.items() if m == other_mother]
            add("connecting_facts", f"Which brothers of {mother}'s son are children of {other_mother}?",
                f"{mother}'s son is {son}. The {'brother is' if len(brothers) == 1 else 'brothers are'} {names(brothers)}.")
    add("corrections", "Why does Sumitra have just one child among Dasharatha's four sons?",
        "She does not have just one. Sumitra has two sons: Lakshmana and Shatrughna.")
    add("corrections", "Is Lakshmana Sumitra's only child in this family?",
        "No, Sumitra's sons are Lakshmana and Shatrughna.")
    return examples, counts


if __name__ == "__main__":
    rows, counts = build_examples()
    print(json.dumps({"counts": counts, "examples": rows}, ensure_ascii=False))
