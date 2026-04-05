"""
test_validator.py
-----------------
Regression tests for validator.validate_steps.

Run with:
    python3 test_validator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from validator import validate_steps


def run(name, steps, equation, expect_idx, expect_msg_contains=None):
    idx, msg = validate_steps(steps, equation)
    passed = idx == expect_idx
    if expect_msg_contains:
        passed = passed and expect_msg_contains.lower() in msg.lower()
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if not passed:
        print(f"       expected idx={expect_idx}  got idx={idx}")
        print(f"       msg: {msg}")
    return passed


def main():
    results = []

    # ------------------------------------------------------------------
    # Word problem intermediate steps (should NOT trigger format error)
    # ------------------------------------------------------------------
    results.append(run(
        "WP: 40 - 14 = x  is an intermediate step",
        ["40 - 14 = x"],
        "40 - 14 = x",
        expect_idx=-2,
        expect_msg_contains="valid so far",
    ))

    results.append(run(
        "WP: 10 + 5 = x  is an intermediate step",
        ["10 + 5 = x"],
        "10 + 5 = x",
        expect_idx=-2,
        expect_msg_contains="valid so far",
    ))

    results.append(run(
        "WP: 4 * 6 = x  is an intermediate step",
        ["4 * 6 = x"],
        "4 * 6 = x",
        expect_idx=-2,
        expect_msg_contains="valid so far",
    ))

    results.append(run(
        "WP: 12 / 6 = x  is an intermediate step",
        ["12 / 6 = x"],
        "12 / 6 = x",
        expect_idx=-2,
        expect_msg_contains="valid so far",
    ))

    # ------------------------------------------------------------------
    # Word problem final solutions
    # ------------------------------------------------------------------
    results.append(run(
        "WP: x = 26 is correct final answer",
        ["x = 26"],
        "40 - 14 = x",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    results.append(run(
        "WP: 26 = x (flipped) is correct final answer",
        ["26 = x"],
        "40 - 14 = x",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    results.append(run(
        "WP: full working — intermediate then solution",
        ["40 - 14 = x", "x = 26"],
        "40 - 14 = x",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    results.append(run(
        "WP: x = 15 for ADD problem",
        ["10 + 5 = x", "x = 15"],
        "10 + 5 = x",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    # ------------------------------------------------------------------
    # Plain linear equation steps
    # ------------------------------------------------------------------
    results.append(run(
        "Linear: intermediate step 2x = 7",
        ["2x = 7"],
        "2x + 3 = 10",
        expect_idx=-2,
        expect_msg_contains="valid so far",
    ))

    results.append(run(
        "Linear: correct solution x = 3.5",
        ["x = 3.5"],
        "2x + 3 = 10",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    results.append(run(
        "Linear: correct integer solution x = 3",
        ["x = 3"],
        "2x + 3 = 9",
        expect_idx=-1,
        expect_msg_contains="correct",
    ))

    results.append(run(
        "Linear: wrong solution flagged",
        ["x = 99"],
        "2x + 3 = 9",
        expect_idx=0,
    ))

    results.append(run(
        "Linear: missing equals sign",
        ["2x 7"],
        "2x + 3 = 10",
        expect_idx=0,
    ))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total  = len(results)
    print(f"\n{passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
