"""
test_word_problem.py
---------------------
Standalone test script for the word problem neuro-symbolic pipeline.

Runs three independent tests:

  Test 1 — PSS generator only
      Verifies the schema generator produces valid, diverse pairs
      without touching any neural network.

  Test 2 — PSS symbolic verifier
      Checks that the numbers embedded in each generated sentence
      are consistent with the schema slots (A, B, answer).

  Test 3 — Trained T5 inference  (only runs if word_problem_model/ exists)
      Loads the fine-tuned model and generates sentences from new schemas
      that were NOT in the training data.

Usage:
    # Generate pairs and inspect (no model needed):
    python3 test_word_problem.py --pss-only

    # Full test including model inference (requires trained model):
    python3 test_word_problem.py
"""

import os
import sys
import json
import argparse
import re

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "word_problem_model")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _pass(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Test 1 — PSS generator
# ---------------------------------------------------------------------------

def test_pss_generator():
    _separator("Test 1: PSS Schema Generator")
    from word_problem_generator import generate_training_pairs, TEMPLATE_SCHEMAS

    pairs = generate_training_pairs(n=500, seed=99)

    if len(pairs) == 500:
        _pass(f"Generated exactly 500 pairs")
    else:
        _fail(f"Expected 500 pairs, got {len(pairs)}")

    template_counts = {}
    for p in pairs:
        template_counts[p["template"]] = template_counts.get(p["template"], 0) + 1

    all_templates_present = set(template_counts.keys()) == set(TEMPLATE_SCHEMAS.keys())
    if all_templates_present:
        _pass("All template types are represented")
    else:
        missing = set(TEMPLATE_SCHEMAS.keys()) - set(template_counts.keys())
        _fail(f"Missing templates: {missing}")

    print("\n  Template distribution:")
    for t, count in sorted(template_counts.items()):
        print(f"    {t:<30} {count} pairs")

    print("\n  Sample pairs:")
    seen = set()
    for p in pairs:
        if p["template"] not in seen:
            seen.add(p["template"])
            print(f"\n    Template : {p['template']}")
            print(f"    Schema   : {p['schema']}")
            print(f"    Sentence : {p['sentence']}")
        if len(seen) == len(TEMPLATE_SCHEMAS):
            break

    return pairs


# ---------------------------------------------------------------------------
# Test 2 — PSS symbolic verifier
# ---------------------------------------------------------------------------

def test_symbolic_verifier(pairs):
    _separator("Test 2: PSS Symbolic Verifier (numbers in sentence match schema)")

    errors = []
    for p in pairs:
        A      = str(p["slots"]["A"])
        B      = str(p["slots"]["B"])
        answer = str(p["slots"]["answer"])
        name   = p["slots"]["name"]
        sent   = p["sentence"]

        if A not in sent:
            errors.append(f"A={A} missing in: {sent}")
        if B not in sent:
            errors.append(f"B={B} missing in: {sent}")
        if name not in sent:
            errors.append(f"name={name} missing in: {sent}")

    if not errors:
        _pass(f"All 500 sentences contain correct A, B, and name values")
    else:
        _fail(f"{len(errors)} sentences failed slot verification")
        for e in errors[:5]:
            print(f"    {e}")

    # Verify equations are algebraically correct
    eq_errors = 0
    for p in pairs:
        A = p["slots"]["A"]
        B = p["slots"]["B"]
        ans = p["slots"]["answer"]
        tmpl = p["template"]
        if tmpl == "ADD_TOTAL"             and A + B != ans: eq_errors += 1
        if tmpl == "SUBTRACT_REMAINING"    and A - B != ans: eq_errors += 1
        if tmpl == "MULTIPLY_TOTAL"        and A * B != ans: eq_errors += 1
        if tmpl == "DIVIDE_SHARE"          and A // B != ans: eq_errors += 1
        if tmpl == "FIND_UNKNOWN_ADD"      and A + ans != B: eq_errors += 1
        if tmpl == "FIND_UNKNOWN_SUBTRACT" and A - ans != B: eq_errors += 1

    if eq_errors == 0:
        _pass("All 500 pairs have algebraically correct answers")
    else:
        _fail(f"{eq_errors} pairs have incorrect answers")


# ---------------------------------------------------------------------------
# Test 3 — T5 inference
# ---------------------------------------------------------------------------

def test_model_inference():
    _separator("Test 3: Fine-tuned T5 Inference")

    if not os.path.isdir(MODEL_DIR):
        print(f"  [SKIP] Model not found at {MODEL_DIR}")
        print(f"         Run train_word_problem_model.py first.")
        return

    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        print("  [SKIP] torch / transformers not installed.")
        return

    print(f"  Loading model from {MODEL_DIR} ...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Unseen schemas — names and objects not heavily represented in training
    test_schemas = [
        "generate problem: template=ADD_TOTAL name=Zara object=shells A=14 B=6 answer=20",
        "generate problem: template=SUBTRACT_REMAINING name=Felix object=buttons A=25 B=9 answer=16",
        "generate problem: template=MULTIPLY_TOTAL name=Priya object=cupcakes A=4 B=7 answer=28",
        "generate problem: template=DIVIDE_SHARE name=Omar object=grapes A=36 B=6 answer=6",
        "generate problem: template=FIND_UNKNOWN_ADD name=Chloe object=ribbons A=8 B=17 answer=9",
        "generate problem: template=FIND_UNKNOWN_SUBTRACT name=Dylan object=badges A=20 B=7 answer=13",
    ]

    print("\n  Generated sentences from unseen schemas:\n")
    all_passed = True
    for schema in test_schemas:
        input_ids = tokenizer(
            schema,
            max_length=128,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract expected numbers from schema
        m_A      = re.search(r"\bA=(\d+)",      schema)
        m_B      = re.search(r"\bB=(\d+)",      schema)
        m_name   = re.search(r"\bname=(\w+)",   schema)
        expected_A    = m_A.group(1)    if m_A    else ""
        expected_B    = m_B.group(1)    if m_B    else ""
        expected_name = m_name.group(1) if m_name else ""

        number_ok = expected_A in sentence and expected_B in sentence
        name_ok   = expected_name in sentence

        status = "PASS" if (number_ok and name_ok) else "WARN"
        if status == "WARN":
            all_passed = False

        print(f"  [{status}] Schema  : {schema}")
        print(f"        Sentence: {sentence}")
        if not number_ok:
            print(f"        WARNING : expected numbers {expected_A}, {expected_B} not found in output")
        if not name_ok:
            print(f"        WARNING : expected name '{expected_name}' not found in output")
        print()

    if all_passed:
        _pass("All inference outputs contain correct numbers and names")
    else:
        print("  [WARN] Some outputs deviated — consider more training epochs or pairs")


# ---------------------------------------------------------------------------
# Single sentence generation — PSS + optional model
# ---------------------------------------------------------------------------

def generate_single(template=None, name=None, obj=None, A=None, B=None, seed=None, use_model=True):
    """
    Generate one word problem sentence.

    If use_model=True and word_problem_model/ exists, the fine-tuned T5 model
    produces the sentence.  Otherwise the PSS slot-fill template is used.

    Parameters can be left as None — the PSS will pick random values.

    Returns a dict with keys: schema, sentence, slots.
    """
    import random as _random
    from word_problem_generator import (
        generate_training_pairs, TEMPLATE_SCHEMAS, SCHEMA_BUILDERS,
        NAMES, OBJECTS, _fill_slots,
    )

    rng = _random.Random(seed)

    template_id = template or rng.choice(list(TEMPLATE_SCHEMAS.keys()))
    numeric     = SCHEMA_BUILDERS[template_id](rng)
    if A is not None: numeric["A"] = A
    if B is not None: numeric["B"] = B
    chosen_name    = name or rng.choice(NAMES)
    obj_s, obj_p   = rng.choice(OBJECTS) if obj is None else (obj, obj + "s")
    sent_tmpl      = rng.choice(TEMPLATE_SCHEMAS[template_id]["sentence_templates"])

    schema_str, pss_sentence = _fill_slots(
        template_id, numeric, chosen_name, obj_s, obj_p, sent_tmpl
    )

    sentence = pss_sentence

    if use_model and os.path.isdir(MODEL_DIR):
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
            model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
            device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            input_ids = tokenizer(schema_str, max_length=128, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
            sentence = tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"  [WARN] Model inference failed ({e}), using PSS sentence.")
            sentence = pss_sentence

    return {
        "schema":   schema_str,
        "sentence": sentence,
        "slots":    {"template": template_id, **numeric, "name": chosen_name, "object": obj_p},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pss-only", action="store_true",
                        help="Run only PSS tests (no model required)")
    parser.add_argument("--single", action="store_true",
                        help="Generate and print one sentence then exit")
    parser.add_argument("--template", type=str, default=None,
                        help="Template ID e.g. ADD_TOTAL, SUBTRACT_REMAINING, MULTIPLY_TOTAL, DIVIDE_SHARE, FIND_UNKNOWN_ADD, FIND_UNKNOWN_SUBTRACT")
    parser.add_argument("--name",   type=str, default=None, help="Person's name")
    parser.add_argument("--object", type=str, default=None, help="Object (singular)")
    parser.add_argument("--A",      type=int, default=None, help="Value for slot A")
    parser.add_argument("--B",      type=int, default=None, help="Value for slot B")
    parser.add_argument("--seed",   type=int, default=None, help="Random seed")
    parser.add_argument("--no-model", action="store_true",
                        help="Use PSS template only, skip model even if available")
    args = parser.parse_args()

    if args.single:
        result = generate_single(
            template=args.template,
            name=args.name,
            obj=args.object,
            A=args.A,
            B=args.B,
            seed=args.seed,
            use_model=not args.no_model,
        )
        print(f"Schema  : {result['schema']}")
        print(f"Sentence: {result['sentence']}")
        print(f"Slots   : {result['slots']}")
    else:
        pairs = test_pss_generator()
        test_symbolic_verifier(pairs)

        if not args.pss_only:
            test_model_inference()
        else:
            print("\n[--pss-only] Skipping model inference test.")

        print("\nDone.")
