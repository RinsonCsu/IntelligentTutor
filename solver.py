import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
)
import heapq
import re

def solve_equation_steps(equation):
    x = sp.symbols('x')
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)

    def _parse(s):
        return parse_expr(s, transformations=transformations)

    lhs, rhs = equation.split("=")
    steps = []
    steps.append(f"Original: {equation}")

    eq = sp.Eq(_parse(lhs), _parse(rhs))
    simplified = sp.simplify(eq)

    expr0 = sp.simplify(simplified.lhs - simplified.rhs)
    try:
        poly = sp.Poly(expr0, x)
    except Exception:
        poly = None

    if poly is not None and poly.degree() == 2:
        std = sp.Eq(sp.expand(expr0), 0)
        steps.append(f"Standard form: {std}")

        content = sp.Integer(1)
        try:
            content, primitive = sp.Poly(std.lhs, x).primitive()
        except Exception:
            primitive = std.lhs

        if content not in (0, 1, -1):
            steps.append(f"GCF: {sp.factor_terms(std.lhs)} = 0")

        factored = sp.factor(std.lhs)

        is_diff_squares = False
        if isinstance(std.lhs, sp.Add) and len(std.lhs.args) == 2:
            a, b = std.lhs.args
            if a.is_Pow and sp.simplify(a.exp - 2) == 0 and (-b).is_Pow and sp.simplify((-b).exp - 2) == 0:
                is_diff_squares = True
            if b.is_Pow and sp.simplify(b.exp - 2) == 0 and (-a).is_Pow and sp.simplify((-a).exp - 2) == 0:
                is_diff_squares = True

        if is_diff_squares:
            steps.append(f"Difference of squares: {factored} = 0")
        else:
            terms = sp.Poly(std.lhs, x).terms()
            if len(terms) == 3:
                steps.append(f"Factoring trinomials: {factored} = 0")
            else:
                steps.append(f"Factor: {factored} = 0")

        solution = sp.solve(std, x)
        steps.append(f"Solve: x = {solution}")
    else:
        steps.append(f"Simplified: {simplified}")
        solution = sp.solve(simplified, x)
        steps.append(f"Solve: x = {solution}")

    return steps, solution

def astar_next_step_isolate(equation, target_var="x", max_expansions=200):
    x, y = sp.symbols('x y')
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)

    def _parse(s):
        return parse_expr(s, transformations=transformations)

    if target_var == "x":
        target = x
        other = y
    elif target_var == "y":
        target = y
        other = x
    else:
        raise ValueError("target_var must be 'x' or 'y'")

    lhs, rhs = equation.split("=")
    start_eq = sp.Eq(sp.expand(_parse(lhs)), sp.expand(_parse(rhs)))

    def _canonical(eq):
        return sp.srepr(eq.lhs), sp.srepr(eq.rhs)

    def _goal(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        if l == target and target not in getattr(r, "free_symbols", set()):
            return True
        if r == target and target not in getattr(l, "free_symbols", set()):
            return True
        return False

    def _heuristic(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        h = 0
        if target in getattr(l, "free_symbols", set()) and target in getattr(r, "free_symbols", set()):
            h += 1

        side_with_target = l if target in getattr(l, "free_symbols", set()) else r
        if target in getattr(side_with_target, "free_symbols", set()):
            try:
                poly = sp.Poly(side_with_target, target)
            except Exception:
                return 10
            if poly is None or poly.degree() > 1:
                return 10
            a = poly.coeffs()[0] if poly.degree() == 1 else 0
            b = poly.TC()
            if sp.simplify(b) != 0:
                h += 1
            if sp.simplify(a - 1) != 0:
                h += 1
        else:
            return 10

        if other in getattr(side_with_target, "free_symbols", set()):
            h += 1

        return int(h)

    def _successors(eq):
        l = sp.expand(eq.lhs)
        r = sp.expand(eq.rhs)
        succ = []

        l_has = target in getattr(l, "free_symbols", set())
        r_has = target in getattr(r, "free_symbols", set())
        if l_has and r_has:
            try:
                pr = sp.Poly(r, target)
            except Exception:
                pr = None
            if pr is not None and pr.degree() == 1:
                a = sp.simplify(pr.coeffs()[0])
                if a != 0:
                    new_l = sp.simplify(l - a * target)
                    new_r = sp.simplify(r - a * target)
                    succ.append((sp.Eq(new_l, new_r), f"subtract {a}*{target} from both sides", 1))

        side_with_target_is_lhs = l_has
        side_with_target = l if side_with_target_is_lhs else r
        other_side = r if side_with_target_is_lhs else l

        try:
            p = sp.Poly(side_with_target, target)
        except Exception:
            p = None

        if p is not None and p.degree() <= 1:
            a = sp.simplify(p.coeffs()[0]) if p.degree() == 1 else sp.Integer(0)
            b = sp.simplify(p.TC())
            if b != 0:
                new_side_with_target = sp.simplify(side_with_target - b)
                new_other_side = sp.simplify(other_side - b)
                if side_with_target_is_lhs:
                    succ.append((sp.Eq(new_side_with_target, new_other_side), f"subtract {b} from both sides", 1))
                else:
                    succ.append((sp.Eq(new_other_side, new_side_with_target), f"subtract {b} from both sides", 1))

            if a != 0 and sp.simplify(a - 1) != 0:
                new_l = sp.simplify(l / a)
                new_r = sp.simplify(r / a)
                # Prefer clearing the constant term before dividing.
                cost = 2 if b != 0 else 1
                succ.append((sp.Eq(new_l, new_r), f"divide both sides by {a}", cost))

        simplified = sp.Eq(sp.simplify(l), sp.simplify(r))
        if simplified != eq:
            succ.append((simplified, "simplify", 1))

        return succ

    start_key = _canonical(start_eq)
    open_heap = []
    heapq.heappush(open_heap, (_heuristic(start_eq), 0, start_key))
    came_from = {start_key: None}
    state_map = {start_key: start_eq}
    g_score = {start_key: 0}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, g, key = heapq.heappop(open_heap)
        current = state_map[key]

        if _goal(current):
            path = [key]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            if len(path) < 2:
                return None
            next_eq = state_map[path[1]]
            return f"{sp.sstr(next_eq.lhs)} = {sp.sstr(next_eq.rhs)}"

        expansions += 1

        for nxt, _, step_cost in _successors(current):
            nxt_key = _canonical(nxt)
            tentative_g = g + step_cost
            if nxt_key not in g_score or tentative_g < g_score[nxt_key]:
                came_from[nxt_key] = key
                state_map[nxt_key] = nxt
                g_score[nxt_key] = tentative_g
                f = tentative_g + _heuristic(nxt)
                heapq.heappush(open_heap, (f, tentative_g, nxt_key))

    return None

def astar_next_step_factor_quadratic(equation, max_expansions=400):
    x = sp.symbols('x')
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)

    def _parse(s):
        return parse_expr(s, transformations=transformations)

    lhs, rhs = equation.split("=")
    start_eq = sp.Eq(_parse(lhs), _parse(rhs))

    def _canonical(eq):
        return sp.srepr(eq.lhs), sp.srepr(eq.rhs)

    def _as_poly0(eq):
        expr = sp.simplify(eq.lhs - eq.rhs)
        try:
            p = sp.Poly(expr, x)
        except Exception:
            return None
        if p is None or p.degree() != 2:
            return None
        return p

    if _as_poly0(start_eq) is None:
        return None

    try:
        if sp.simplify(start_eq.rhs) == 0:
            from expert_system import infer_expert_next_step as _expert_next
            expert_result = _expert_next(equation)
            if expert_result is not None:
                return expert_result[0]
    except Exception:
        pass

    def _goal(eq):
        if sp.simplify(eq.rhs) != 0:
            return False
        lhs_s = sp.simplify(eq.lhs)
        if isinstance(lhs_s, sp.Add):
            return False

        factors = sp.factor(lhs_s)
        if isinstance(factors, sp.Mul):
            facs = list(factors.args)
        else:
            facs = [factors]

        has_linear = False
        for f in facs:
            if x not in getattr(f, "free_symbols", set()):
                continue
            try:
                p = sp.Poly(f, x)
            except Exception:
                return False
            if p is None or p.degree() != 1:
                return False
            has_linear = True
        return has_linear

    def _heuristic(eq):
        h = 0
        if sp.simplify(eq.rhs) != 0:
            h += 2
        if sp.simplify(eq.rhs) == 0:
            if _goal(eq):
                h += 0
            else:
                h += 2
        return h

    def _successors(eq):
        l = sp.expand(eq.lhs)
        r = sp.expand(eq.rhs)
        succ = []

        def _is_split_middle_form(expr):
            if not isinstance(expr, sp.Add):
                return False
            if len(expr.args) != 4:
                return False
            try:
                x_terms = [
                    t
                    for t in expr.args
                    if x in getattr(t, "free_symbols", set()) and sp.Poly(t, x).degree() == 1
                ]
                x2_terms = [
                    t
                    for t in expr.args
                    if x in getattr(t, "free_symbols", set()) and sp.Poly(t, x).degree() == 2
                ]
            except Exception:
                return False
            const_terms = [t for t in expr.args if x not in getattr(t, "free_symbols", set())]
            return len(x_terms) == 2 and len(x2_terms) == 1 and len(const_terms) == 1

        if sp.simplify(r) != 0:
            succ.append((sp.Eq(sp.expand(l - r), 0), "move to 0", 1))

        if sp.simplify(eq.rhs) == 0:
            expr = sp.expand(eq.lhs)
            try:
                p = sp.Poly(expr, x)
            except Exception:
                p = None
            if p is not None and p.degree() == 2:
                a, b, c = p.all_coeffs()
                g = sp.gcd(a, sp.gcd(b, c))
                if g not in (0, 1, -1):
                    succ.append((sp.Eq(sp.factor_terms(expr), 0), "GCF", 1))

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
                                succ.append((sp.Eq(split_expr, 0), "Factoring trinomials", 1))
                                break

            fac = sp.factor(expr)
            if fac != sp.simplify(expr):
                cost = 1 if _is_split_middle_form(eq.lhs) else 5
                succ.append((sp.Eq(fac, 0), "Factoring trinomials", cost))

        simplified = sp.Eq(sp.expand(eq.lhs), sp.expand(eq.rhs))
        if simplified != eq:
            succ.append((simplified, "simplify", 1))

        return succ

    start_key = _canonical(start_eq)
    open_heap = []
    heapq.heappush(open_heap, (_heuristic(start_eq), 0, start_key))
    came_from = {start_key: None}
    state_map = {start_key: start_eq}
    g_score = {start_key: 0}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, g, key = heapq.heappop(open_heap)
        current = state_map[key]

        if _goal(current):
            path = [key]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            if len(path) < 2:
                return None
            next_eq = state_map[path[1]]
            return f"{sp.sstr(next_eq.lhs)} = {sp.sstr(next_eq.rhs)}"

        expansions += 1

        for nxt, _, step_cost in _successors(current):
            nxt_key = _canonical(nxt)
            tentative_g = g + step_cost
            if nxt_key not in g_score or tentative_g < g_score[nxt_key]:
                came_from[nxt_key] = key
                state_map[nxt_key] = nxt
                g_score[nxt_key] = tentative_g
                f = tentative_g + _heuristic(nxt)
                heapq.heappush(open_heap, (f, tentative_g, nxt_key))

    return None

def astar_next_step_complete_square(equation, max_expansions=400):
    x = sp.symbols('x')
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application,)

    def _parse(s):
        return parse_expr(s, transformations=transformations)

    lhs, rhs = equation.split("=")
    start_eq = sp.Eq(sp.simplify(_parse(lhs)), sp.simplify(_parse(rhs)))

    def _canonical(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        return sp.srepr(l), sp.srepr(r)

    def _is_quadratic_eq(eq):
        expr = sp.simplify(eq.lhs - eq.rhs)
        try:
            p = sp.Poly(expr, x)
        except Exception:
            return False
        return p is not None and p.degree() == 2

    if not _is_quadratic_eq(start_eq):
        return None

    def _goal(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        if x in getattr(r, "free_symbols", set()):
            return False
        if isinstance(l, sp.Pow) and sp.simplify(l.exp - 2) == 0:
            base = sp.simplify(l.base)
            try:
                p = sp.Poly(base, x)
            except Exception:
                return False
            return p is not None and p.degree() == 1
        return False

    def _heuristic(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        expr = sp.simplify(l - r)
        try:
            poly = sp.Poly(expr, x)
        except Exception:
            return 10
        if poly is None or poly.degree() != 2:
            if _goal(eq):
                return 0
            return 10

        a = sp.simplify(poly.coeffs()[0])
        b = sp.simplify(poly.coeffs()[1])
        c = sp.simplify(poly.TC())

        h = 0
        if sp.simplify(a - 1) != 0:
            h += 1
        if c != 0 and sp.simplify(r) == 0:
            h += 1
        if not (isinstance(sp.simplify(l), sp.Pow) and sp.simplify(sp.simplify(l).exp - 2) == 0):
            h += 1
        if b != 0:
            h += 1
        return int(min(10, h))

    def _successors(eq):
        l = sp.simplify(eq.lhs)
        r = sp.simplify(eq.rhs)
        succ = []

        expr = sp.simplify(l - r)
        try:
            poly = sp.Poly(expr, x)
        except Exception:
            poly = None
        if poly is None or poly.degree() != 2:
            return succ

        a = sp.simplify(poly.coeffs()[0])
        b = sp.simplify(poly.coeffs()[1])
        c = sp.simplify(poly.TC())

        if sp.simplify(eq.rhs) == 0:
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
                            succ.append((sp.Eq(a * x**2 + m * x + n * x + c, 0), "split middle term"))
                            break

        std_expr = sp.expand(a * x**2 + b * x + c)
        if not (sp.simplify(r) == 0 and sp.simplify(l - std_expr) == 0):
            succ.append((sp.Eq(std_expr, 0), "standard form"))

        if a != 0 and sp.simplify(a - 1) != 0 and sp.simplify(r) == 0:
            succ.append((sp.Eq(sp.simplify(std_expr / a), 0), f"divide by {a}"))

        expr2 = sp.simplify(eq.lhs - eq.rhs)
        try:
            poly2 = sp.Poly(expr2, x)
        except Exception:
            poly2 = None
        if poly2 is None or poly2.degree() != 2:
            return succ
        a2 = sp.simplify(poly2.coeffs()[0])
        b2 = sp.simplify(poly2.coeffs()[1])
        c2 = sp.simplify(poly2.TC())

        if sp.simplify(eq.rhs) == 0:
            if c2 != 0:
                succ.append((sp.Eq(sp.expand(a2 * x**2 + b2 * x), sp.expand(-c2)), "move constant"))

        if x in getattr(eq.rhs, "free_symbols", set()):
            return succ

        if sp.simplify(eq.lhs - (a2 * x**2 + b2 * x)) == 0:
            if a2 != 0:
                add_term = sp.simplify((b2 / (2 * a2))**2)
                succ.append((sp.Eq(sp.simplify(eq.lhs + add_term), sp.simplify(eq.rhs + add_term)), "complete square"))

        if x not in getattr(eq.rhs, "free_symbols", set()):
            try:
                fact_lhs = sp.factor(eq.lhs)
            except Exception:
                fact_lhs = eq.lhs
            if isinstance(fact_lhs, sp.Pow) and sp.simplify(fact_lhs.exp - 2) == 0:
                base = sp.simplify(fact_lhs.base)
                try:
                    base_poly = sp.Poly(base, x)
                except Exception:
                    base_poly = None
                if base_poly is not None and base_poly.degree() == 1 and fact_lhs != eq.lhs:
                    succ.append((sp.Eq(fact_lhs, sp.simplify(eq.rhs)), "rewrite square"))

        simplified = sp.Eq(sp.expand(eq.lhs), sp.expand(eq.rhs))
        if simplified != eq:
            succ.append((simplified, "simplify"))

        return succ

    start_key = _canonical(start_eq)
    open_heap = []
    heapq.heappush(open_heap, (_heuristic(start_eq), 0, start_key))
    came_from = {start_key: None}
    state_map = {start_key: start_eq}
    g_score = {start_key: 0}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, g, key = heapq.heappop(open_heap)
        current = state_map[key]

        if _goal(current):
            path = [key]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            if len(path) < 2:
                return None
            next_eq = state_map[path[1]]
            return f"{sp.simplify(next_eq.lhs)} = {sp.simplify(next_eq.rhs)}"

        expansions += 1

        for nxt, _ in _successors(current):
            nxt_key = _canonical(nxt)
            tentative_g = g + 1
            if nxt_key not in g_score or tentative_g < g_score[nxt_key]:
                came_from[nxt_key] = key
                state_map[nxt_key] = nxt
                g_score[nxt_key] = tentative_g
                f = tentative_g + _heuristic(nxt)
                heapq.heappush(open_heap, (f, tentative_g, nxt_key))

    return None
