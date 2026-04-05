"""
word_problem_generator.py
--------------------------
Physical Symbol System (PSS) based word problem schema generator.

Generates (schema_string, sentence) training pairs for fine-tuning T5-small.

The PSS owns all mathematical structure:
  - template type   (ADD, SUBTRACT, MULTIPLY, DIVIDE)
  - numeric slots   (A, B, answer)
  - semantic slots  (name, object, verb_phrase)

The sentence side is produced by slot-filling a curated set of hand-written
sentence templates — no neural network is involved here.  These pairs become
the supervised training data for train_word_problem_model.py.
"""

import random
import json
import os

# ---------------------------------------------------------------------------
# PSS Symbol store — named slot vocabulary
# ---------------------------------------------------------------------------

NAMES = [
    "John", "Maria", "Sarah", "Tom", "Emma", "Liam", "Olivia", "Noah",
    "Ava", "James", "Sophia", "Lucas", "Mia", "Ethan", "Isabella",
    "Mason", "Charlotte", "Logan", "Amelia", "Jacob",
]

OBJECTS = [
    ("pencil",   "pencils"),
    ("apple",    "apples"),
    ("book",     "books"),
    ("cookie",   "cookies"),
    ("marble",   "marbles"),
    ("sticker",  "stickers"),
    ("coin",     "coins"),
    ("balloon",  "balloons"),
    ("crayon",   "crayons"),
    ("card",     "cards"),
    ("orange",   "oranges"),
    ("stamp",    "stamps"),
    ("toy",      "toys"),
    ("bottle",   "bottles"),
    ("flower",   "flowers"),
]

# ---------------------------------------------------------------------------
# PSS Template schemas — one per equation type
# Sentence templates use {name}, {A}, {B}, {answer}, {obj}, {objs} as slots
# ---------------------------------------------------------------------------

TEMPLATE_SCHEMAS = {

    "ADD_TOTAL": {
        "equation": "x = A + B",
        "sentence_templates": [
            "{name} has {A} {objs}. They got {B} more. How many {objs} do they have in total?",
            "{name} had {A} {objs} and received {B} more. What is the total number of {objs}?",
            "{name} sees {A} {objs} in a box and adds {B} more {objs}. How many are there now?",
            "{name} collected {A} {objs}. Later, they found {B} more. How many {objs} do they have altogether?",
            "If {name} starts with {A} {objs} and gets {B} more, what is the total?",
        ],
    },

    "SUBTRACT_REMAINING": {
        "equation": "x = A - B",
        "sentence_templates": [
            "{name} has {A} {objs} and gave away {B}. How many {objs} are left?",
            "{name} started with {A} {objs} but lost {B} of them. How many remain?",
            "{name} had {A} {objs} on the shelf and took {B} away. How many are left?",
            "{name} had {A} {objs}. After using {B}, how many {objs} does {name} have?",
            "If {name} has {A} {objs} and gives {B} to a friend, how many are remaining?",
        ],
    },

    "MULTIPLY_TOTAL": {
        "equation": "x = A * B",
        "sentence_templates": [
            "{name} has {A} bags with {B} {objs} in each. How many {objs} in total?",
            "{name} counts {A} boxes, each containing {B} {objs}. How many {objs} are there altogether?",
            "{name} packs {B} {objs} into each of {A} bags. What is the total number of {objs}?",
            "If {name} places {B} {objs} on each of {A} tables, how many {objs} are placed in total?",
            "{name} buys {A} packs of {objs}, each containing {B}. How many {objs} does {name} have?",
        ],
    },

    "DIVIDE_SHARE": {
        "equation": "x = A / B",
        "sentence_templates": [
            "{name} has {A} {objs} and shares them equally among {B} friends. How many does each friend get?",
            "{name} divides {A} {objs} equally into {B} groups. How many are in each group?",
            "{name} wants to put {A} {objs} into {B} equal piles. How many {objs} per pile?",
            "{name} shares {A} {objs} equally between {B} people. How many does each person receive?",
            "{name} divides {A} {objs} equally among {B} students. How many {objs} does each student get?",
        ],
    },

    "FIND_UNKNOWN_ADD": {
        "equation": "A + x = B",
        "sentence_templates": [
            "{name} has {A} {objs}. After receiving some more, they have {B}. How many did they receive?",
            "{name} counts {A} {objs} in a jar. More are added to make {B} total. How many were added?",
            "{name} started with {A} {objs} and now has {B}. How many {objs} were added?",
            "{name} has a basket with {A} {objs}. After adding more it contains {B}. How many {objs} were put in?",
            "{name} had {A} {objs} and bought some more, ending up with {B}. How many were bought?",
        ],
    },

    "FIND_UNKNOWN_SUBTRACT": {
        "equation": "A - x = B",
        "sentence_templates": [
            "{name} had {A} {objs} and gave some away, leaving {B}. How many were given away?",
            "{name} had {A} {objs}. Some were removed and {B} remained. How many were removed?",
            "{name} started with {A} {objs}. After losing some, {B} are left. How many were lost?",
            "{name} started with {A} {objs}. Some were used up and only {B} remain. How many were used?",
            "{name} had {A} {objs}. They gave some to a friend and now have {B}. How many did they give?",
        ],
    },
}

# ---------------------------------------------------------------------------
# PSS production rules — numeric slot instantiation
# ---------------------------------------------------------------------------

def _build_add_schema(rng):
    A = rng.randint(2, 30)
    B = rng.randint(1, 20)
    return {"A": A, "B": B, "answer": A + B}

def _build_subtract_schema(rng):
    B = rng.randint(1, 20)
    A = rng.randint(B + 1, B + 30)
    return {"A": A, "B": B, "answer": A - B}

def _build_multiply_schema(rng):
    A = rng.randint(2, 12)
    B = rng.randint(2, 10)
    return {"A": A, "B": B, "answer": A * B}

def _build_divide_schema(rng):
    B = rng.randint(2, 10)
    answer = rng.randint(2, 12)
    A = B * answer
    return {"A": A, "B": B, "answer": answer}

def _build_find_unknown_add_schema(rng):
    A = rng.randint(1, 20)
    x = rng.randint(1, 20)
    B = A + x
    return {"A": A, "B": B, "answer": x}

def _build_find_unknown_subtract_schema(rng):
    x = rng.randint(1, 20)
    B = rng.randint(1, 15)
    A = B + x
    return {"A": A, "B": B, "answer": x}


SCHEMA_BUILDERS = {
    "ADD_TOTAL":              _build_add_schema,
    "SUBTRACT_REMAINING":     _build_subtract_schema,
    "MULTIPLY_TOTAL":         _build_multiply_schema,
    "DIVIDE_SHARE":           _build_divide_schema,
    "FIND_UNKNOWN_ADD":       _build_find_unknown_add_schema,
    "FIND_UNKNOWN_SUBTRACT":  _build_find_unknown_subtract_schema,
}

# ---------------------------------------------------------------------------
# Slot filler — composes schema into input/output pair
# ---------------------------------------------------------------------------

def _fill_slots(template_id, numeric, name, obj_singular, obj_plural, sentence_template):
    sentence = sentence_template.format(
        name=name,
        A=numeric["A"],
        B=numeric["B"],
        answer=numeric["answer"],
        obj=obj_singular,
        objs=obj_plural,
    )
    schema_str = (
        f"generate problem: template={template_id} "
        f"name={name} object={obj_plural} "
        f"A={numeric['A']} B={numeric['B']} answer={numeric['answer']}"
    )
    return schema_str, sentence


# ---------------------------------------------------------------------------
# Generator — produces N training pairs
# ---------------------------------------------------------------------------

def generate_training_pairs(n: int = 500, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    template_ids = list(TEMPLATE_SCHEMAS.keys())

    while len(pairs) < n:
        template_id = rng.choice(template_ids)
        numeric     = SCHEMA_BUILDERS[template_id](rng)
        name        = rng.choice(NAMES)
        obj_s, obj_p = rng.choice(OBJECTS)
        sent_tmpl   = rng.choice(TEMPLATE_SCHEMAS[template_id]["sentence_templates"])

        schema_str, sentence = _fill_slots(
            template_id, numeric, name, obj_s, obj_p, sent_tmpl
        )

        pairs.append({
            "template":  template_id,
            "equation":  TEMPLATE_SCHEMAS[template_id]["equation"],
            "schema":    schema_str,
            "sentence":  sentence,
            "slots": {
                "name":   name,
                "object": obj_p,
                "A":      numeric["A"],
                "B":      numeric["B"],
                "answer": numeric["answer"],
            },
        })

    return pairs


def _equation_str(template_id: str, numeric: dict) -> str:
    """Return a solver-compatible equation string for the given template and slots."""
    A = numeric["A"]
    B = numeric["B"]
    ans = numeric["answer"]
    mapping = {
        "ADD_TOTAL":              f"{A} + {B} = x",
        "SUBTRACT_REMAINING":     f"{A} - {B} = x",
        "MULTIPLY_TOTAL":         f"{A} * {B} = x",
        "DIVIDE_SHARE":           f"{A} / {B} = x",
        "FIND_UNKNOWN_ADD":       f"{A} + x = {B}",
        "FIND_UNKNOWN_SUBTRACT":  f"{A} - x = {B}",
    }
    return mapping.get(template_id, f"x = {ans}")


def build_word_problem(template_id: str = None, seed: int = None) -> tuple:
    """
    Generate one word problem using PSS slot-filling (no model required).

    Returns:
        (sentence, equation_str)
        - sentence     : human-readable word problem string
        - equation_str : solver-compatible equation e.g. "10 + 5 = x"
    """
    rng = random.Random(seed)
    tid = template_id or rng.choice(list(TEMPLATE_SCHEMAS.keys()))
    numeric  = SCHEMA_BUILDERS[tid](rng)
    name     = rng.choice(NAMES)
    obj_s, obj_p = rng.choice(OBJECTS)
    tmpl     = rng.choice(TEMPLATE_SCHEMAS[tid]["sentence_templates"])
    _, sentence = _fill_slots(tid, numeric, name, obj_s, obj_p, tmpl)
    eq_str   = _equation_str(tid, numeric)
    return sentence, eq_str


def save_pairs(pairs: list[dict], path: str):
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {path}")


def load_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point — generate and save pairs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "word_problem_pairs.json")
    pairs = generate_training_pairs(n=500, seed=42)
    save_pairs(pairs, out_path)

    print("\nSample pairs:")
    for p in pairs[:3]:
        print(f"\n  Schema  : {p['schema']}")
        print(f"  Sentence: {p['sentence']}")
