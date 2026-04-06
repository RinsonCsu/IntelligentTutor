import sympy as sp
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
)

from expert_system import (
    expert_solution_format_feedback,
    expert_missing_equals_feedback,
    expert_multiple_equals_feedback,
    expert_incorrect_solution_feedback,
    expert_not_equivalent_feedback,
    expert_progress_feedback,
    expert_all_correct_feedback,
    expert_is_acceptable_final_answer,
)

x = sp.symbols('x')
transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)

def _parse(s):
    return parse_expr(s, transformations=transformations)

_PLAIN_NUMBER_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

def _is_plain_number_str(s: str) -> bool:
    """True only if s is a bare numeric literal — no operators or expressions."""
    return bool(_PLAIN_NUMBER_RE.match(s.strip().replace(" ", "")))

def _extract_solution_value(eq, lhs_str: str = "", rhs_str: str = ""):
    if eq.lhs == x and x not in getattr(eq.rhs, "free_symbols", set()):
        if _is_plain_number_str(rhs_str):
            return sp.simplify(eq.rhs)
    if eq.rhs == x and x not in getattr(eq.lhs, "free_symbols", set()):
        if _is_plain_number_str(lhs_str):
            return sp.simplify(eq.lhs)
    return None

def _estimate_linear_steps(eq):
    lhs_s = sp.simplify(eq.lhs)
    rhs_s = sp.simplify(eq.rhs)

    if lhs_s == x and x not in getattr(rhs_s, "free_symbols", set()):
        return 0 if rhs_s.is_Atom else 1
    if rhs_s == x and x not in getattr(lhs_s, "free_symbols", set()):
        return 0 if lhs_s.is_Atom else 1

    def _lin_parts(side):
        try:
            p = sp.Poly(side, x)
        except Exception:
            return None
        if p is None or p.degree() > 1:
            return None
        a = sp.simplify(p.coeffs()[0]) if p.degree() == 1 else sp.Integer(0)
        b = sp.simplify(p.TC())
        return a, b

    lhs_parts = _lin_parts(lhs_s)
    rhs_parts = _lin_parts(rhs_s)
    if lhs_parts is None or rhs_parts is None:
        return None

    a_l, b_l = lhs_parts
    a_r, b_r = rhs_parts

    steps = 0

    if a_l != 0 and a_r != 0:
        steps += 1
        a = sp.simplify(a_l - a_r)
        b_left = sp.simplify(b_l - b_r)

        if b_left != 0:
            steps += 1

        if a not in (0, 1):
            steps += 1
        return steps

    if a_l != 0:
        a = a_l
        x_side_const = b_l
    elif a_r != 0:
        a = a_r
        x_side_const = b_r
    else:
        return None

    if x_side_const != 0:
        steps += 1

    if a not in (0, 1):
        steps += 1

    return steps

def estimate_linear_steps_remaining(equation, current_step):
    lhs, rhs = equation.split("=")
    original = sp.Eq(_parse(lhs), _parse(rhs))

    if isinstance(current_step, str):
        if current_step.count("=") != 1:
            return None, None
        l2, r2 = current_step.split("=")
        current_eq = sp.Eq(_parse(l2), _parse(r2))
    else:
        current_eq = current_step

    start = _estimate_linear_steps(original)
    cur = _estimate_linear_steps(current_eq)
    return start, cur

def solve_linear_equation_value(equation):
    lhs, rhs = equation.split("=")
    eq = sp.Eq(_parse(lhs), _parse(rhs))
    try:
        sol = sp.solve(eq, x)
    except Exception:
        return None
    if not sol:
        return None
    if len(sol) != 1:
        return None
    val = sol[0]
    if x in getattr(val, "free_symbols", set()):
        return None
    return sp.simplify(val)

def validate_steps(student_steps, equation):
    lhs, rhs = equation.split("=")
    original = sp.Eq(_parse(lhs), _parse(rhs))
    prev = original
    last_step_str = ""
    found_solution_line = False

    def _validate_solution_value(sol_value):
        residual = sp.simplify(original.lhs.subs(x, sol_value) - original.rhs.subs(x, sol_value))
        return residual == 0

    def _is_acceptable_final_answer_text(s):
        return expert_is_acceptable_final_answer(s)

    def _total_expected_lines():
        """Estimate total lines a student needs to write."""
        ops = _estimate_linear_steps(original)
        if ops is None:
            return None
        lhs_s = sp.simplify(original.lhs)
        rhs_s = sp.simplify(original.rhs)
        try:
            lp = sp.Poly(lhs_s, x)
            rp = sp.Poly(rhs_s, x)
            a_l = lp.coeffs()[0] if lp.degree() == 1 else sp.Integer(0)
            a_r = rp.coeffs()[0] if rp.degree() == 1 else sp.Integer(0)
            x_on_both = (a_l != 0 and a_r != 0)
            net_coeff = sp.simplify(a_l - a_r)
        except Exception:
            x_on_both = False
            net_coeff = sp.Integer(1)
        # When x is on both sides, _estimate_linear_steps already counts the
        # division step internally — do not double-count it.
        # When x is on one side only, add 1 if coefficient != ±1.
        if x_on_both:
            has_division = False
        else:
            has_division = net_coeff not in (sp.Integer(1), sp.Integer(-1))
        return ops + (1 if has_division else 0)

    def _progress_percent(lines_done):
        total = _total_expected_lines()
        if total is None or total <= 0:
            return None
        pct = int(round(100 * lines_done / total))
        return max(0, min(99, pct))

    for i, step in enumerate(student_steps):
        try:
            if "=" not in step:
                return i, expert_missing_equals_feedback()

            clauses = [step]
            if re.search(r"\bor\b", step, flags=re.IGNORECASE):
                clauses = [c.strip() for c in re.split(r"\bor\b", step, flags=re.IGNORECASE) if c.strip()]

            all_clauses_are_solutions = True
            all_clauses_have_acceptable_format = True
            for clause in clauses:
                if clause.count("=") != 1:
                    all_clauses_are_solutions = False
                    break
                l, r = clause.split("=")
                eq_clause = sp.Eq(_parse(l), _parse(r))
                sol_value = _extract_solution_value(eq_clause, lhs_str=l, rhs_str=r)
                if sol_value is None:
                    all_clauses_are_solutions = False
                    break
                sol_text = l if _is_plain_number_str(l) else r
                if not _is_acceptable_final_answer_text(sol_text):
                    all_clauses_have_acceptable_format = False
                if not _validate_solution_value(sol_value):
                    return i, expert_incorrect_solution_feedback()

            if all_clauses_are_solutions and all_clauses_have_acceptable_format:
                found_solution_line = True
                continue

            if all_clauses_are_solutions and not all_clauses_have_acceptable_format:
                rhs_texts = [c.split("=")[1].strip() for c in clauses if c.count("=") == 1]
                msg = expert_solution_format_feedback(rhs_texts)
                return -2, msg or "Correct solutions, but format final answers as a decimal (>= 2 dp) or a proper fraction"

            if len(clauses) > 1:
                clause_eqs = []
                for clause in clauses:
                    if clause.count("=") != 1:
                        return i, expert_multiple_equals_feedback()
                    cl, cr = clause.split("=")
                    try:
                        clause_eqs.append(sp.Eq(_parse(cl), _parse(cr)))
                    except Exception:
                        return i, expert_multiple_equals_feedback()
                prev_expr = sp.simplify(prev.lhs - prev.rhs)
                try:
                    prev_factors = sp.factor(prev_expr)
                    if isinstance(prev_factors, sp.Mul):
                        factor_list = [f for f in prev_factors.args if x in getattr(f, "free_symbols", set())]
                    else:
                        factor_list = [prev_factors] if x in getattr(prev_factors, "free_symbols", set()) else []
                except Exception:
                    factor_list = []
                valid_split = False
                if factor_list and len(factor_list) == len(clause_eqs):
                    matched = [False] * len(factor_list)
                    all_matched = True
                    for ceq in clause_eqs:
                        ceq_expr = sp.simplify(ceq.lhs - ceq.rhs)
                        found = False
                        for fi, fac in enumerate(factor_list):
                            if not matched[fi]:
                                ratio = sp.simplify(fac / ceq_expr) if ceq_expr != 0 else None
                                if ratio is not None and getattr(ratio, "free_symbols", set()) == set() and ratio != 0:
                                    matched[fi] = True
                                    found = True
                                    break
                        if not found:
                            all_matched = False
                            break
                    valid_split = all_matched
                if not valid_split:
                    return i, expert_not_equivalent_feedback()
                prev = clause_eqs[-1]
                continue

            if step.count("=") != 1:
                return i, expert_multiple_equals_feedback()

            l, r = step.split("=")
            current = sp.Eq(_parse(l), _parse(r))

            prev_expr = sp.simplify(prev.lhs - prev.rhs)
            curr_expr = sp.simplify(current.lhs - current.rhs)

            equivalent = sp.simplify(prev_expr - curr_expr) == 0
            if not equivalent:
                if curr_expr != 0:
                    ratio = sp.simplify(prev_expr / curr_expr)
                    if ratio != 0 and getattr(ratio, "free_symbols", set()) == set():
                        equivalent = sp.simplify(prev_expr - ratio * curr_expr) == 0

            if not equivalent:
                return i, expert_not_equivalent_feedback()

            prev = current
            last_step_str = step

        except Exception as e:
            return i, f"Invalid expression ({type(e).__name__}): {e}"

    if found_solution_line:
        return -1, expert_all_correct_feedback()

    pct = _progress_percent(len(student_steps))
    return -2, expert_progress_feedback(pct)
