"""
word_problem_generator.py
--------------------------
Neuro-symbolic word problem generator.

The PSS owns all mathematical structure:
  - template type   (ADD, SUBTRACT, MULTIPLY, DIVIDE)
  - numeric slots   (A, B, answer)
  - semantic slots  (name, object, verb_phrase)

If the fine-tuned T5 model exists in word_problem_model/, it is loaded once
at import time and used to generate sentences from PSS schemas.  If not, the
PSS slot-fill templates are used as a fallback.

The equation string always comes from the PSS regardless of which path
generates the sentence, guaranteeing mathematical correctness.
"""

import random
import json
import os

# ---------------------------------------------------------------------------
# T5 model — loaded once at import time if word_problem_model/ exists
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "word_problem_model")
_t5_model     = None
_t5_tokenizer = None
_t5_device    = None

def _load_t5_model():
    global _t5_model, _t5_tokenizer, _t5_device
    if not os.path.isdir(_MODEL_DIR):
        return
    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _t5_tokenizer = T5Tokenizer.from_pretrained(_MODEL_DIR)
        _t5_model     = T5ForConditionalGeneration.from_pretrained(_MODEL_DIR)
        _t5_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _t5_model.to(_t5_device)
        _t5_model.eval()
        print(f"[word_problem_generator] T5 model loaded from {_MODEL_DIR}")
    except Exception as e:
        print(f"[word_problem_generator] Could not load T5 model ({e}), using PSS templates.")
        _t5_model = None

_load_t5_model()

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
    A = B + x          # guarantees A > B always
    assert A > B, f"Schema error: A={A} must be > B={B}"
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


def _t5_generate(schema_str: str) -> str:
    """Use the fine-tuned T5 model to generate a sentence from a schema string."""
    import torch
    input_ids = _t5_tokenizer(
        schema_str, max_length=128, return_tensors="pt"
    ).input_ids.to(_t5_device)
    with torch.no_grad():
        out = _t5_model.generate(
            input_ids, max_length=128, num_beams=4, early_stopping=True
        )
    return _t5_tokenizer.decode(out[0], skip_special_tokens=True)


def build_word_problem(template_id: str = None, seed: int = None) -> tuple:
    """
    Generate one word problem.

    If the fine-tuned T5 model is loaded, it generates the sentence.
    Otherwise the PSS slot-fill template is used as fallback.
    The equation string always comes from the PSS.

    Returns:
        (sentence, equation_str)
        - sentence     : human-readable word problem string
        - equation_str : solver-compatible equation e.g. "10 + 5 = x"
    """
    rng = random.Random(seed)
    tid      = template_id or rng.choice(list(TEMPLATE_SCHEMAS.keys()))
    numeric  = SCHEMA_BUILDERS[tid](rng)
    name     = rng.choice(NAMES)
    obj_s, obj_p = rng.choice(OBJECTS)
    tmpl     = rng.choice(TEMPLATE_SCHEMAS[tid]["sentence_templates"])
    schema_str, pss_sentence = _fill_slots(tid, numeric, name, obj_s, obj_p, tmpl)
    eq_str   = _equation_str(tid, numeric)

    import re as _re

    def _numbers_valid(sent, num):
        """All expected numbers appear in the sentence."""
        for n in (num["A"], num["B"]):
            if not _re.search(r'\b' + str(n) + r'\b', sent):
                return False
        return True

    def _semantically_valid(sent, num, template):
        """For subtract templates, starting number must appear before remainder."""
        if template in ("FIND_UNKNOWN_SUBTRACT", "SUBTRACT_REMAINING"):
            def _pos(s, n):
                m = _re.search(r'\b' + str(n) + r'\b', s)
                return m.start() if m else -1
            if _pos(sent, num["A"]) > _pos(sent, num["B"]):
                return False
        # Universal: detect "had X ... now have Y" where Y > X (impossible subtract story)
        had = _re.search(r'\bhad\s+(\d+)\b', sent)
        now = _re.search(r'\bnow\s+have\s+(\d+)\b', sent)
        if had and now and int(now.group(1)) > int(had.group(1)):
            return False
        return True

    if _t5_model is not None:
        max_retries = 5
        sentence = None
        for attempt in range(max_retries):
            try:
                attempt_rng = random.Random(seed + attempt * 1000 if seed is not None else attempt)
                attempt_numeric = SCHEMA_BUILDERS[tid](attempt_rng) if attempt > 0 else numeric
                attempt_name = attempt_rng.choice(NAMES) if attempt > 0 else name
                attempt_obj_s, attempt_obj_p = attempt_rng.choice(OBJECTS) if attempt > 0 else (obj_s, obj_p)
                attempt_tmpl = attempt_rng.choice(TEMPLATE_SCHEMAS[tid]["sentence_templates"]) if attempt > 0 else tmpl
                attempt_schema, attempt_pss = _fill_slots(
                    tid, attempt_numeric, attempt_name, attempt_obj_s, attempt_obj_p, attempt_tmpl
                )
                candidate = _t5_generate(attempt_schema)
                if _numbers_valid(candidate, attempt_numeric) and _semantically_valid(candidate, attempt_numeric, tid):
                    # Accept — update outputs if we used a different attempt
                    if attempt > 0:
                        numeric.update(attempt_numeric)
                        eq_str = _equation_str(tid, attempt_numeric)
                        pss_sentence = attempt_pss
                    sentence = candidate
                    break
            except Exception as e:
                print(f"[word_problem_generator] T5 attempt {attempt+1} failed ({e}).")
        if sentence is None:
            raise ValueError(
                f"[word_problem_generator] T5 failed to produce a valid sentence "
                f"for template='{tid}' after {max_retries} attempts. "
                f"Last schema: {schema_str}"
            )
    else:
        sentence = pss_sentence

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
