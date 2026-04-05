"""
symbolic_planner.py
-------------------
Physical Symbol System (PSS) based equation planner.

A PSS represents knowledge as symbolic structures and manipulates them via
explicit symbol operations (Newell & Simon, 1976).  Here, equations are built
from a small vocabulary of named symbols and rewrite rules that compose them
into well-formed linear expressions of increasing difficulty.

Difficulty is parameterised by a DifficultyProfile (linear),
WordProblemProfile (word problem), or QuadraticProfile (quadratic), which controls:
  - coefficient magnitude
  - constant magnitude
  - presence of x on both sides  (linear)
  - word problem template type   (word problem)
  - factor root range and sign    (quadratic)
  - multi-step requirements (number of operations to solve)
"""

import random
import sympy as sp
from word_problem_generator import build_word_problem

x = sp.Symbol("x")

# ---------------------------------------------------------------------------
# Symbol vocabulary (PSS symbol store)
# ---------------------------------------------------------------------------

SYMBOL_TYPES = {
    "VAR":        "x",
    "COEFF":      None,   # filled at generation time
    "CONST":      None,
    "RHS":        None,
    "FACTOR_P":   None,   # root of first linear factor  (x + p)
    "FACTOR_Q":   None,   # root of second linear factor (x + q)
    "QUAD_CONST": None,   # optional RHS shift for quadratic
    "WP_TEMPLATE": None, # word problem template type
    "WP_NAME":    None,   # name slot for word problem
    "WP_OBJECT":  None,   # object slot for word problem
}

# ---------------------------------------------------------------------------
# Difficulty profiles  (PSS rule templates)
# ---------------------------------------------------------------------------

class DifficultyProfile:
    def __init__(self,
                 level: int,
                 coeff_range: tuple,
                 const_range: tuple,
                 rhs_range: tuple,
                 allow_negative_coeff: bool = False,
                 allow_negative_const: bool = False,
                 x_on_both_sides: bool = False,
                 rhs_coeff_range: tuple = (1, 1)):
        self.level              = level
        self.coeff_range        = coeff_range
        self.const_range        = const_range
        self.rhs_range          = rhs_range
        self.allow_negative_coeff = allow_negative_coeff
        self.allow_negative_const = allow_negative_const
        self.x_on_both_sides    = x_on_both_sides
        self.rhs_coeff_range    = rhs_coeff_range


# ---------------------------------------------------------------------------
# Word problem profile  (PSS rule template for word problems)
# ---------------------------------------------------------------------------

class WordProblemProfile:
    """Rule template for word problems built by PSS slot-filling.

    The PSS production rule picks a template type, then fills name, object,
    and numeric slots to produce a sentence + solver-compatible equation.
    """

    def __init__(self, level: int, template_id: str):
        self.level       = level
        self.template_id = template_id  # e.g. "ADD_TOTAL"


# ---------------------------------------------------------------------------
# Quadratic profile  (PSS rule template for factor-first construction)
# ---------------------------------------------------------------------------

class QuadraticProfile:
    """Rule template for quadratic equations built by choosing roots first.

    The PSS production rule picks FACTOR_P and FACTOR_Q as the roots of two
    linear factors, multiplies them symbolically to get ax^2 + bx + c, then
    optionally shifts the RHS away from zero so the student must rearrange.
    """

    def __init__(self,
                 level: int,
                 root_range: tuple,
                 allow_negative_roots: bool = False,
                 nonzero_rhs: bool = False,
                 rhs_shift_range: tuple = (1, 5)):
        self.level               = level
        self.root_range          = root_range
        self.allow_negative_roots = allow_negative_roots
        self.nonzero_rhs         = nonzero_rhs
        self.rhs_shift_range     = rhs_shift_range


DIFFICULTY_PROFILES = [
    DifficultyProfile(
        level=1,
        coeff_range=(2, 4),
        const_range=(1, 5),
        rhs_range=(5, 15),
        allow_negative_coeff=False,
        allow_negative_const=False,
        x_on_both_sides=False,
    ),
    DifficultyProfile(
        level=2,
        coeff_range=(2, 6),
        const_range=(1, 10),
        rhs_range=(5, 25),
        allow_negative_coeff=False,
        allow_negative_const=True,
        x_on_both_sides=False,
    ),
    DifficultyProfile(
        level=3,
        coeff_range=(3, 8),
        const_range=(2, 12),
        rhs_range=(5, 30),
        allow_negative_coeff=True,
        allow_negative_const=True,
        x_on_both_sides=False,
    ),
    DifficultyProfile(
        level=4,
        coeff_range=(3, 10),
        const_range=(2, 15),
        rhs_range=(1, 10),
        allow_negative_coeff=True,
        allow_negative_const=True,
        x_on_both_sides=True,
        rhs_coeff_range=(1, 3),
    ),
    DifficultyProfile(
        level=5,
        coeff_range=(4, 12),
        const_range=(3, 20),
        rhs_range=(1, 10),
        allow_negative_coeff=True,
        allow_negative_const=True,
        x_on_both_sides=True,
        rhs_coeff_range=(2, 5),
    ),
    WordProblemProfile(level=6,  template_id="ADD_TOTAL"),
    WordProblemProfile(level=7,  template_id="SUBTRACT_REMAINING"),
    WordProblemProfile(level=8,  template_id="FIND_UNKNOWN_ADD"),
    WordProblemProfile(level=9,  template_id="FIND_UNKNOWN_SUBTRACT"),
    WordProblemProfile(level=10, template_id="MULTIPLY_TOTAL"),
    WordProblemProfile(level=11, template_id="DIVIDE_SHARE"),
    QuadraticProfile(
        level=12,
        root_range=(1, 9),
        allow_negative_roots=False,
        nonzero_rhs=False,
    ),
    QuadraticProfile(
        level=13,
        root_range=(1, 9),
        allow_negative_roots=True,
        nonzero_rhs=True,
        rhs_shift_range=(1, 9),
    ),
]


# ---------------------------------------------------------------------------
# PSS rewrite rules  (symbol manipulation operations)
# ---------------------------------------------------------------------------

def _pick(lo, hi, allow_negative, rng):
    """Pick a non-zero integer in [lo, hi], optionally negated."""
    val = rng.randint(lo, hi)
    if allow_negative and rng.random() < 0.5:
        val = -val
    return val


def _render_side(coeff, var, const):
    """Render  coeff*x + const  as a clean human-readable string."""
    parts = []
    if coeff == 1:
        parts.append("x")
    elif coeff == -1:
        parts.append("-x")
    else:
        parts.append(f"{coeff}x")

    if const > 0:
        parts.append(f"+ {const}")
    elif const < 0:
        parts.append(f"- {abs(const)}")

    return " ".join(parts)


def _build_equation(profile: DifficultyProfile, rng: random.Random) -> str:
    """
    PSS production rule: instantiate symbols from the profile's template
    and compose them into a verified linear equation string.
    """
    for _ in range(200):
        a = _pick(*profile.coeff_range, profile.allow_negative_coeff, rng)
        b = _pick(*profile.const_range, profile.allow_negative_const, rng)

        if profile.x_on_both_sides:
            c = _pick(*profile.rhs_coeff_range, False, rng)
            d = _pick(*profile.rhs_range, profile.allow_negative_const, rng)
            lhs_expr = a * x + b
            rhs_expr = c * x + d
            net_coeff = a - c
            if net_coeff == 0:
                continue
            sol = sp.Rational(d - b, net_coeff)
            lhs_str = _render_side(a, "x", b)
            rhs_str = _render_side(c, "x", d)
        else:
            rhs_val = _pick(*profile.rhs_range, False, rng)
            sol = sp.Rational(rhs_val - b, a)
            lhs_str = _render_side(a, "x", b)
            rhs_str = str(rhs_val)

        if sol.denominator != 1:
            continue
        if sol == 0:
            continue

        return f"{lhs_str} = {rhs_str}"

    return _fallback(profile.level)


def _render_quadratic(a, b, c) -> str:
    """Render ax^2 + bx + c as a clean human-readable string."""
    parts = []
    if a == 1:
        parts.append("x^2")
    elif a == -1:
        parts.append("-x^2")
    else:
        parts.append(f"{a}x^2")

    if b == 1:
        parts.append("+ x")
    elif b == -1:
        parts.append("- x")
    elif b > 0:
        parts.append(f"+ {b}x")
    elif b < 0:
        parts.append(f"- {abs(b)}x")

    if c > 0:
        parts.append(f"+ {c}")
    elif c < 0:
        parts.append(f"- {abs(c)}")

    return " ".join(parts)


def _build_quadratic_equation(profile: QuadraticProfile, rng: random.Random) -> str:
    """
    PSS production rule for quadratic equations.

    Symbol instantiation sequence:
      1. Bind FACTOR_P  <- random single-digit integer (root of first factor)
      2. Bind FACTOR_Q  <- random single-digit integer (root of second factor)
      3. Compose (x + FACTOR_P)(x + FACTOR_Q) symbolically via SymPy
      4. Expand to standard form ax^2 + bx + c
      5. Optionally bind QUAD_CONST and shift RHS to make rearrangement required
    """
    for _ in range(200):
        p = _pick(*profile.root_range, profile.allow_negative_roots, rng)
        q = _pick(*profile.root_range, profile.allow_negative_roots, rng)
        if p == q:
            continue

        factor1 = x + p
        factor2 = x + q
        expanded = sp.expand(factor1 * factor2)

        try:
            poly = sp.Poly(expanded, x)
            a, b, c = [int(v) for v in poly.all_coeffs()]
        except Exception:
            continue

        if not profile.nonzero_rhs:
            lhs_str = _render_quadratic(a, b, c)
            return f"{lhs_str} = 0"
        else:
            shift = rng.randint(*profile.rhs_shift_range)
            if rng.random() < 0.5:
                shift = -shift
            new_c = c - shift
            lhs_str = _render_quadratic(a, b, new_c)
            return f"{lhs_str} = {-shift}"

    return _fallback(profile.level)


def _build_word_problem_equation(profile: WordProblemProfile, rng: random.Random) -> str:
    """
    PSS production rule for word problems.

    Symbol instantiation sequence:
      1. Bind WP_TEMPLATE  <- profile.template_id
      2. Bind WP_NAME      <- random name from vocabulary
      3. Bind WP_OBJECT    <- random object from vocabulary
      4. Fill numeric slots A, B via SCHEMA_BUILDER
      5. Render sentence via slot-fill template
      6. Append solver-compatible equation string
    """
    try:
        seed = rng.randint(0, 999999)
        sentence, eq_str = build_word_problem(
            template_id=profile.template_id,
            seed=seed,
        )
        return f"{sentence}\n[equation: {eq_str}]"
    except Exception:
        return _fallback(profile.level)


def _fallback(level: int) -> str:
    """Hardcoded safe fallbacks per difficulty level."""
    fallbacks = {
        1:  "2x + 3 = 11",
        2:  "3x - 4 = 14",
        3:  "-2x + 5 = 13",
        4:  "4x + 3 = 2x + 11",
        5:  "5x - 7 = 2x + 8",
        6:  "John has 10 apples. They got 5 more. How many apples do they have in total?\n[equation: 10 + 5 = x]",
        7:  "Maria had 20 pencils and gave away 8. How many pencils are left?\n[equation: 20 - 8 = x]",
        8:  "Tom has 6 cookies. After receiving some more, they have 14. How many did they receive?\n[equation: 6 + x = 14]",
        9:  "Emma had 15 books and gave some away, leaving 9. How many were given away?\n[equation: 15 - x = 9]",
        10: "Sarah has 4 bags with 6 apples in each. How many apples in total?\n[equation: 4 * 6 = x]",
        11: "Liam has 24 coins and shares them equally among 6 friends. How many does each friend get?\n[equation: 24 / 6 = x]",
        12: "x^2 + 5x + 6 = 0",
        13: "x^2 + 2x - 3 = 4",
    }
    return fallbacks.get(level, "2x + 1 = 9")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SymbolicPlanner:
    """
    Stateful PSS planner.  Each call to next_equation() advances to the next
    difficulty level and returns a freshly generated equation guaranteed to be
    strictly harder than the previous one.
    """

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)
        self._level_index = 0
        self._history: list[str] = []

    @property
    def current_level(self) -> int:
        return DIFFICULTY_PROFILES[self._level_index].level

    def current_equation(self) -> str:
        """Return the most recently generated equation without advancing."""
        if self._history:
            return self._history[-1]
        return self._generate_current()

    def _generate_current(self) -> str:
        eq = self._dispatch(DIFFICULTY_PROFILES[self._level_index])
        self._history.append(eq)
        return eq

    def _dispatch(self, profile) -> str:
        """Route to the correct PSS production rule based on profile type."""
        if isinstance(profile, QuadraticProfile):
            return _build_quadratic_equation(profile, self._rng)
        if isinstance(profile, WordProblemProfile):
            return _build_word_problem_equation(profile, self._rng)
        return _build_equation(profile, self._rng)

    def next_equation(self) -> str:
        """
        Advance to the next difficulty level (capped at the hardest profile)
        and return a new equation.
        """
        if self._level_index < len(DIFFICULTY_PROFILES) - 1:
            self._level_index += 1
        eq = self._dispatch(DIFFICULTY_PROFILES[self._level_index])
        self._history.append(eq)
        return eq

    def reset(self, seed: int = None):
        """Reset the planner to level 1 with a fresh random seed."""
        self._rng = random.Random(seed)
        self._level_index = 0
        self._history.clear()
