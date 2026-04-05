import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
)


x = sp.symbols("x")
transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)


def _parse(s: str):
    return parse_expr(s, transformations=transformations)


def _parse_eq(equation_text: str):
    if "=" not in equation_text:
        return None
    if equation_text.count("=") != 1:
        return None
    lhs, rhs = equation_text.split("=")
    try:
        return sp.Eq(sp.expand(_parse(lhs)), sp.expand(_parse(rhs)))
    except Exception:
        return None


def _eq_to_text(eq: sp.Eq):
    return f"{sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}"


def _is_trivial_same_hint(before: sp.Eq, after: sp.Eq):
    try:
        b = sp.simplify(before.lhs - before.rhs)
        a = sp.simplify(after.lhs - after.rhs)
    except Exception:
        return True
    if sp.simplify(b - a) == 0:
        return True
    if a != 0 and sp.simplify(b / a + 1) == 0:
        return True
    return False


def infer_expert_next_step(equation_text: str):
    eq = _parse_eq(equation_text)
    if eq is None:
        return None

    expr0 = sp.simplify(eq.lhs - eq.rhs)
    try:
        poly = sp.Poly(expr0, x)
    except Exception:
        poly = None

    if poly is not None and poly.degree() == 2:
        if sp.simplify(eq.rhs) != 0:
            nxt = sp.Eq(sp.expand(eq.lhs - eq.rhs), 0)
            if _is_trivial_same_hint(eq, nxt):
                return None
            return (
                _eq_to_text(nxt),
                "Rule: STANDARD_FORM. Move everything to one side to get = 0.",
            )

        a, b, c = poly.all_coeffs()
        if isinstance(a, sp.Integer) and isinstance(b, sp.Integer) and isinstance(c, sp.Integer):
            ac = int(a * c)
            if ac != 0:
                candidates = []
                for d in range(1, abs(ac) + 1):
                    if ac % d != 0:
                        continue
                    candidates.append(d)
                    candidates.append(-d)
                for m_int in candidates:
                    n_int = ac // m_int
                    if m_int + n_int == int(b):
                        m = sp.Integer(m_int)
                        n = sp.Integer(n_int)
                        split_expr = sp.Add(a * x**2, m * x, n * x, c, evaluate=False)
                        nxt = sp.Eq(split_expr, 0)
                        if _is_trivial_same_hint(eq, nxt):
                            return None
                        return (
                            _eq_to_text(nxt),
                            "Rule: SPLIT_MIDDLE_TERM. Rewrite bx as mx + nx to factor by grouping.",
                        )

        return None

    try:
        lhs_p = sp.Poly(eq.lhs, x)
        rhs_p = sp.Poly(eq.rhs, x)
    except Exception:
        return None

    if lhs_p is None or rhs_p is None:
        return None
    if lhs_p.degree() > 1 or rhs_p.degree() > 1:
        return None

    if sp.simplify(eq.lhs - eq.rhs) == 0:
        return None

    if lhs_p.degree() == 1 and rhs_p.degree() == 0:
        a = sp.simplify(lhs_p.coeffs()[0])
        b = sp.simplify(lhs_p.TC())
        if b != 0:
            nxt = sp.Eq(sp.expand(eq.lhs - b), sp.expand(eq.rhs - b))
            if _is_trivial_same_hint(eq, nxt):
                return None
            return (
                _eq_to_text(nxt),
                "Rule: MOVE_CONSTANT. Subtract the constant term from both sides.",
            )
        if a != 0 and sp.simplify(a - 1) != 0:
            nxt = sp.Eq(sp.expand(eq.lhs / a), sp.expand(eq.rhs / a))
            if _is_trivial_same_hint(eq, nxt):
                return None
            return (
                _eq_to_text(nxt),
                "Rule: DIVIDE_COEFFICIENT. Divide both sides by the coefficient of x.",
            )

    if rhs_p.degree() == 1 and lhs_p.degree() == 0:
        a = sp.simplify(rhs_p.coeffs()[0])
        b = sp.simplify(rhs_p.TC())
        if b != 0:
            nxt = sp.Eq(sp.expand(eq.lhs - b), sp.expand(eq.rhs - b))
            if _is_trivial_same_hint(eq, nxt):
                return None
            return (
                _eq_to_text(nxt),
                "Rule: MOVE_CONSTANT. Subtract the constant term from both sides.",
            )
        if a != 0 and sp.simplify(a - 1) != 0:
            nxt = sp.Eq(sp.expand(eq.lhs / a), sp.expand(eq.rhs / a))
            if _is_trivial_same_hint(eq, nxt):
                return None
            return (
                _eq_to_text(nxt),
                "Rule: DIVIDE_COEFFICIENT. Divide both sides by the coefficient of x.",
            )

    return None


def diagnose_step_error(equation_text: str, prev_step_text: str, current_step_text: str, validator_message: str):
    original = _parse_eq(equation_text)
    prev_eq = _parse_eq(prev_step_text)
    cur_eq = _parse_eq(current_step_text)

    if validator_message.startswith("Missing '") or "Missing '=" in validator_message:
        return ("Invalid", "Rule: MISSING_EQUALS. Each step must contain exactly one '='.")

    if original is None or prev_eq is None or cur_eq is None:
        return None

    if validator_message == "Not equivalent transformation":
        if sp.simplify(prev_eq.lhs - cur_eq.lhs) == 0 and sp.simplify(prev_eq.rhs - cur_eq.rhs) != 0:
            return (
                "Algebraic Error",
                "Rule: BALANCE_ERROR. You changed only one side of the equation; do the same operation to both sides.",
            )
        if sp.simplify(prev_eq.rhs - cur_eq.rhs) == 0 and sp.simplify(prev_eq.lhs - cur_eq.lhs) != 0:
            return (
                "Algebraic Error",
                "Rule: BALANCE_ERROR. You changed only one side of the equation; do the same operation to both sides.",
            )
        try:
            prev_expr = sp.simplify(prev_eq.lhs - prev_eq.rhs)
            cur_expr = sp.simplify(cur_eq.lhs - cur_eq.rhs)
        except Exception:
            prev_expr = None
            cur_expr = None
        if prev_expr is not None and cur_expr is not None:
            if sp.simplify(prev_expr + cur_expr) == 0:
                return (
                    "Sign Error",
                    "Rule: SIGN_FLIP. It looks like the whole equation was multiplied by -1 unintentionally.",
                )

        return (
            "Algebraic Error",
            "Rule: NOT_EQUIVALENT. This step is not algebraically equivalent to the previous one.",
        )

    if validator_message == "Incorrect solution":
        return (
            "Arithmetic Error",
            "Rule: CHECK_SOLUTION. Substitute your value back into the original equation to verify it.",
        )

    if validator_message.startswith("Invalid expression"):
        return (
            "Invalid",
            "Rule: PARSE_ERROR. The expression could not be parsed; check parentheses, operators, and syntax.",
        )

    return None


def expert_solution_format_feedback(rhs_texts):
    if not rhs_texts:
        return None
    return "Correct solutions, but format final answers as a decimal (>= 2 dp) or a proper fraction"


def expert_missing_equals_feedback():
    return "Rule: MISSING_EQUALS. Each step must contain exactly one '='."


def expert_multiple_equals_feedback():
    return "Rule: MULTIPLE_EQUALS. Invalid expression; each step should have exactly one '='."


def expert_incorrect_solution_feedback():
    return "Rule: CHECK_SOLUTION. Incorrect solution; substitute your value back into the original equation to verify."


def expert_not_equivalent_feedback():
    return "Rule: NOT_EQUIVALENT. This step is not algebraically equivalent to the previous one."


def expert_progress_feedback(pct):
    if pct is None:
        return "Valid so far (keep going)"
    return f"Valid so far: about {pct}% of the way there"


def expert_all_correct_feedback():
    return "All correct"


def expert_is_acceptable_final_answer(s):
    import re
    s = s.strip().replace(" ", "")
    m = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return False
        return abs(num) < abs(den)
    if re.fullmatch(r"[+-]?\d+\.\d{2,}", s):
        return True
    return False
