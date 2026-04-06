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
    # SOLVE_ROOTS: handle "x+p=0 or x+q=0" form (contains " or ")
    if " or " in equation_text.lower():
        import re as _re_or
        parts = _re_or.split(r'\bor\b', equation_text, flags=_re_or.IGNORECASE)
        solutions = []
        valid = True
        for part in parts:
            sub_eq = _parse_eq(part.strip())
            if sub_eq is None:
                valid = False
                break
            sols = sp.solve(sub_eq, x)
            if not sols:
                valid = False
                break
            solutions.append(sols[0])
        if valid and solutions:
            # Check not already solved (all parts already "x = val")
            already_solved = all(
                _re_or.match(r'\s*x\s*=\s*[+-]?\d', p.strip())
                for p in parts
            )
            if not already_solved:
                sol_parts = [f"x = {sp.sstr(s)}" for s in solutions]
                return (
                    " or ".join(sol_parts),
                    "Rule: SOLVE_ROOTS. Solve each linear equation to find the roots.",
                )
        return None

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

        # --- detect which quadratic sub-stage the student is at ---
        import re as _re
        lhs_str = equation_text.split("=")[0]

        # FACTOR_PRODUCT: detect grouped form x(x+m) + n(x+m) = 0
        # Matches: x(expr) + n(expr) or x(expr) - n(expr)
        grp = _re.search(
            r'x\s*\*?\s*\(([^)]+)\)\s*([+-])\s*(\d+)\s*\*?\s*\(([^)]+)\)',
            lhs_str
        )
        if grp:
            inner1 = grp.group(1).strip()
            sign   = grp.group(2)
            n_str  = grp.group(3).strip()
            inner2 = grp.group(4).strip()
            if inner1 == inner2:
                try:
                    inner_expr = _parse(inner1)   # e.g. x+3  (already a binomial)
                    n_val      = _parse(sign + n_str)   # e.g. +6 or -6
                    # factored form: (x + n_val)*(inner_expr) = 0
                    factored = sp.Eq(sp.Mul(x + n_val, inner_expr, evaluate=False), 0)
                    return (
                        _eq_to_text(factored),
                        "Rule: FACTOR_PRODUCT. Extract the common binomial factor to write as a product.",
                    )
                except Exception:
                    pass

        # ZERO_PRODUCT: detect factored form (x+p)(x+q) = 0
        fct = _re.search(r'\(([^)]+)\)\s*\*?\s*\(([^)]+)\)', lhs_str)
        if fct:
            try:
                f1 = _parse(fct.group(1))
                f2 = _parse(fct.group(2))
                r1 = sp.solve(f1, x)
                r2 = sp.solve(f2, x)
                if r1 and r2:
                    def _root_eq(r):
                        c = -r
                        if c > 0:   return f"x + {c} = 0"
                        elif c < 0: return f"x - {-c} = 0"
                        else:       return "x = 0"
                    hint = f"{_root_eq(r1[0])} or {_root_eq(r2[0])}"
                    return (hint, "Rule: ZERO_PRODUCT. If A×B=0 then A=0 or B=0.")
            except Exception:
                pass

        # SPLIT_MIDDLE_TERM or FACTOR_BY_GROUPING
        a, b, c = poly.all_coeffs()
        if isinstance(a, sp.Integer) and isinstance(b, sp.Integer) and isinstance(c, sp.Integer):
            ac    = int(a * c)
            b_int = int(b)
            if ac != 0:
                # Find m, n such that m+n=b and m*n=ac
                pair = None
                for d in range(1, abs(ac) + 1):
                    if ac % d == 0:
                        for sign in (1, -1):
                            m_int = sign * d
                            n_int = ac // m_int
                            if m_int + n_int == b_int:
                                pair = (m_int, n_int)
                                break
                    if pair:
                        break

                if pair:
                    m_int, n_int = pair
                    # Detect split by counting linear x-terms in original string.
                    # Remove x**2 / x^2 tokens, then count remaining 'x' occurrences.
                    lhs_no_sq = _re.sub(r'x\s*(\*\*|[\^])\s*2', '', lhs_str)
                    linear_x_count = len(_re.findall(r'x', lhs_no_sq))
                    if linear_x_count >= 2:
                        # Already split — suggest FACTOR_BY_GROUPING
                        # Build: x(x + m_int/a) + (n_int/a)(x + m_int/a) = 0
                        m_a = sp.Rational(m_int, int(a))
                        n_a = sp.Rational(n_int, int(a))
                        inner = x + m_a
                        grouped = sp.Eq(
                            sp.Mul(x, inner, evaluate=False) + sp.Mul(n_a, inner, evaluate=False),
                            0
                        )
                        return (
                            _eq_to_text(grouped),
                            "Rule: FACTOR_BY_GROUPING. Group the first two and last two terms and factor each group.",
                        )
                    else:
                        # Still unsplit — suggest SPLIT_MIDDLE_TERM
                        m = sp.Integer(m_int)
                        n = sp.Integer(n_int)
                        split_expr = sp.Add(a * x**2, m * x, n * x, c, evaluate=False)
                        nxt = sp.Eq(split_expr, 0)
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

    # Determine which side holds x (or both)
    a_l = lhs_p.coeffs()[0] if lhs_p.degree() == 1 else sp.Integer(0)
    a_r = rhs_p.coeffs()[0] if rhs_p.degree() == 1 else sp.Integer(0)
    b_l = lhs_p.TC()
    b_r = rhs_p.TC()

    # Rule: COLLECT_X — x appears on both sides; move x terms to the left
    if a_l != 0 and a_r != 0:
        net_a = sp.simplify(a_l - a_r)
        net_b = sp.simplify(b_l - b_r)
        if net_b != 0:
            nxt = sp.Eq(net_a * x + net_b, sp.Integer(0))
        else:
            nxt = sp.Eq(net_a * x, sp.Integer(0))
        return (_eq_to_text(nxt), "Rule: COLLECT_X. Subtract x terms from both sides to collect x on one side.")

    # Normalise so x is always on lhs
    if a_r != 0 and a_l == 0:
        eq = sp.Eq(eq.rhs, eq.lhs)
        a_l, a_r = a_r, a_l
        b_l, b_r = b_r, b_l

    # Now: lhs has x, rhs is constant
    # Rule: MOVE_CONSTANT — constant on the same side as x
    if b_l != 0:
        nxt = sp.Eq(a_l * x, sp.expand(b_r - b_l))
        return (_eq_to_text(nxt), "Rule: MOVE_CONSTANT. Subtract the constant from both sides to isolate the x term.")

    # Rule: DIVIDE_COEFFICIENT — coefficient != ±1
    if sp.simplify(a_l - 1) != 0 and sp.simplify(a_l + 1) != 0:
        val = sp.Rational(b_r, a_l)
        nxt = sp.Eq(x, val)
        return (_eq_to_text(nxt), "Rule: DIVIDE_COEFFICIENT. Divide both sides by the coefficient of x.")

    return None


def diagnose_step_error(equation_text: str, prev_step_text: str, current_step_text: str, validator_message: str):
    original = _parse_eq(equation_text)
    prev_eq = _parse_eq(prev_step_text)
    cur_eq = _parse_eq(current_step_text)

    if validator_message.startswith("Missing '") or "Missing '=" in validator_message:
        return ("Invalid", "Rule: MISSING_EQUALS. Each step must contain exactly one '='.")

    if "MULTIPLE_EQUALS" in validator_message:
        import re as _re
        if _re.search(r"\bor\b", current_step_text, flags=_re.IGNORECASE):
            return (
                "Invalid",
                "Rule: ZERO_PRODUCT_SPLIT. When writing two equations from a factored form, "
                "write each as a separate equation: e.g. 'x + 3 = 0 or x + 1 = 0'.",
            )
        return ("Invalid", "Rule: MULTIPLE_EQUALS. Each step must contain exactly one '='.")

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
    if re.fullmatch(r"[+-]?\d+", s):
        return True
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        return True
    m = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return False
        return abs(num) < abs(den)
    return False


def count_steps_remaining(equation_text: str, max_iter: int = 10) -> int | None:
    """
    Simulate expert rules forward from equation_text until no more rules fire.
    Returns the number of lines a student still needs to write.
    When DIVIDE_COEFFICIENT produces a non-integer result, two lines are expected
    (x = a/b then x = value), so that step counts as 2.
    """
    current = equation_text.strip()
    total = 0
    for _ in range(max_iter):
        result = infer_expert_next_step(current)
        if result is None:
            return total
        next_eq, rule = result
        if "DIVIDE_COEFFICIENT" in rule:
            # Check from the CURRENT equation whether a/b simplifies to integer.
            # SymPy auto-simplifies in next_eq so we must inspect before the step.
            cur_eq = _parse_eq(current)
            if cur_eq is not None:
                try:
                    lp = sp.Poly(cur_eq.lhs, x)
                    rp = sp.Poly(cur_eq.rhs, x)
                    a = lp.coeffs()[0] if lp.degree() == 1 else rp.coeffs()[0]
                    b = rp.TC() if lp.degree() == 1 else lp.TC()
                    if sp.simplify(sp.Rational(b, a)).q != 1:
                        total += 1   # extra line for x = a/b before x = value
                except Exception:
                    pass
        total += 1
        current = next_eq
    return total
