# Intelligent Math Teacher

A neuro-symbolic intelligent tutoring system for linear and quadratic equation solving.

---

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | App entry point |
| `gui.py` | Tkinter GUI |
| `expert_system.py` | PSS rule-based expert feedback |
| `validator.py` | Step-by-step algebraic validation |
| `solver.py` | A* equation solver |
| `symbolic_planner.py` | PSS-based random equation generator (levels 1–7) |
| `adaptive.py` | Student model and difficulty adaptation |
| `model.py` | Error classifier |
| `hints.py` | Hint generator |
| `word_problem_generator.py` | PSS word problem schema generator (500 training pairs) |
| `train_word_problem_model.py` | Fine-tunes T5-small on word problem pairs |
| `test_word_problem.py` | Standalone tests for the word problem pipeline |

---

## Running the App

```bash
python3 main.py
```

The app starts with a randomly generated linear equation (Level 1).
Click **Next** to advance to a harder equation — difficulty escalates across 7 levels:

| Level | Type | Example |
|---|---|---|
| 1 | Linear, positive | `3x + 2 = 11` |
| 2 | Linear, negative constants | `4x - 4 = 8` |
| 3 | Linear, negative coefficients | `-5x - 12 = 28` |
| 4 | Linear, x both sides | `7x + 12 = 2x + 7` |
| 5 | Linear, x both sides (larger) | `5x - 17 = 4x - 10` |
| 6 | Quadratic, positive roots | `x^2 + 4x + 3 = 0` |
| 7 | Quadratic, mixed roots, non-zero RHS | `x^2 - 3x - 56 = -2` |

---

## Word Problem Pipeline (standalone, not yet in main app)

### Step 1 — Generate training pairs (PSS only)

```bash
python3 word_problem_generator.py
```

Generates `word_problem_pairs.json` — 500 `(schema → sentence)` pairs across 6 equation template types.

---

### Step 2 — Train the T5-small model

```bash
python3 train_word_problem_model.py
```

- Downloads T5-small (~240 MB, once only)
- Fine-tunes for 5 epochs on the 500 pairs
- Saves the best model to `word_problem_model/`
- **Will not retrain** if `word_problem_model/` already exists

To force a retrain:

```bash
python3 train_word_problem_model.py --force
```

---

### Step 3 — Test the pipeline

**Run all tests (PSS + model inference):**
```bash
python3 test_word_problem.py
```

**Run PSS tests only (no model needed):**
```bash
python3 test_word_problem.py --pss-only
```

---

### Generate a single sentence

**Using the trained T5 model** (falls back to PSS if model not trained):
```bash
python3 test_word_problem.py --single
```

**Using PSS template only** (instant, no model needed):
```bash
python3 test_word_problem.py --single --no-model
```

**With specific values:**
```bash
python3 test_word_problem.py --single --no-model --template ADD_TOTAL --name Alice --A 12 --B 5
```

**Available `--template` values:**
- `ADD_TOTAL`
- `SUBTRACT_REMAINING`
- `MULTIPLY_TOTAL`
- `DIVIDE_SHARE`
- `FIND_UNKNOWN_ADD`
- `FIND_UNKNOWN_SUBTRACT`

**All `--single` options:**

| Flag | Description |
|---|---|
| `--template` | Template type (see above) |
| `--name` | Person's name in the problem |
| `--object` | Object (singular, e.g. `apple`) |
| `--A` | Value for slot A |
| `--B` | Value for slot B |
| `--seed` | Random seed for reproducibility |
| `--no-model` | Use PSS template only, skip T5 model |

---

## Requirements

```bash
python3 -m pip install torch transformers sentencepiece sympy
```
