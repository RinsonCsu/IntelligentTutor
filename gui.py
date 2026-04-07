import tkinter as tk
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
from solver import (
    solve_equation_steps,
    astar_next_step_isolate,
    astar_next_step_complete_square,
    astar_next_step_factor_quadratic,
)
from validator import validate_steps
from adaptive import StudentModel
from symbolic_planner import SymbolicPlanner

_SP_TF = standard_transformations + (convert_xor, implicit_multiplication_application,)

def _student_has_equation_step(steps, eq):
    try:
        eq_parsed = sp.Eq(parse_expr(eq.split("=")[0], transformations=_SP_TF),
                          parse_expr(eq.split("=")[1], transformations=_SP_TF))
    except Exception:
        return False
    for s in steps:
        s = s.strip()
        if not s or "=" not in s:
            continue
        try:
            sp_parsed = sp.Eq(parse_expr(s.split("=")[0], transformations=_SP_TF),
                              parse_expr(s.split("=")[1], transformations=_SP_TF))
            if (sp.simplify(eq_parsed.lhs - sp_parsed.lhs) == 0 and
                    sp.simplify(eq_parsed.rhs - sp_parsed.rhs) == 0):
                return True
            if (sp.simplify(eq_parsed.lhs - sp_parsed.rhs) == 0 and
                    sp.simplify(eq_parsed.rhs - sp_parsed.lhs) == 0):
                return True
        except Exception:
            continue
    return False

class MathTutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Math Teacher")

        FONT      = ("Comic Sans MS", 14)
        FONT_BOLD = ("Comic Sans MS", 14, "bold")
        BG_MAIN   = "#d0e8f5"
        BG_QBOX   = "#1a4a7a"
        BG_OUTPUT = "#ffffff"
        FG_WHITE  = "#ffffff"
        FG_BLACK  = "#000000"
        BTN_DARK  = "#1a3a6e"
        BTN_HOVER = "#2a5aae"
        BTN_DIS   = "#7a8a9e"

        def _make_btn(parent, text, command, **kwargs):
            lbl = tk.Label(parent, text=text, bg=BTN_DARK, fg=FG_WHITE,
                           font=FONT_BOLD, padx=14, pady=7, cursor="hand2",
                           relief=tk.FLAT, **kwargs)
            lbl.bind("<Button-1>", lambda e: command() if lbl.cget("bg") != BTN_DIS else None)
            lbl.bind("<Enter>",    lambda e: lbl.config(bg=BTN_HOVER) if lbl.cget("bg") != BTN_DIS else None)
            lbl.bind("<Leave>",    lambda e: lbl.config(bg=BTN_DARK)  if lbl.cget("bg") != BTN_DIS else None)
            return lbl

        root.configure(bg=BG_MAIN)
        root.minsize(720, 700)

        self.student = StudentModel()
        self.planner = SymbolicPlanner()
        self._question_num = 1
        self._current_equation = ""
        self._wp_equation_hint_shown = False
        self._correct_count = 0
        self._total_count = 0
        self._last_answer_correct = True

        header_frame = tk.Frame(root, bg=BG_MAIN)
        header_frame.pack(fill=tk.X, padx=16, pady=(14, 6))
        tk.Label(header_frame, text="Intelligent Math Teacher",
                 font=("Comic Sans MS", 20, "bold"),
                 bg=BG_MAIN, fg="#1a1a5e").pack(side=tk.LEFT, expand=True)
        self.score_var = tk.StringVar(value="Score: 0 / 0")
        tk.Label(header_frame, textvariable=self.score_var,
                 font=FONT_BOLD, bg=BG_MAIN, fg="#007a00").pack(side=tk.RIGHT)

        self.question_label_var = tk.StringVar(value="1. Question")
        tk.Label(root, textvariable=self.question_label_var,
                 font=FONT_BOLD, bg=BG_MAIN, fg="#1a1a5e").pack(anchor=tk.W, padx=14, pady=(6, 0))

        self.question_box = tk.Text(
            root, height=5, width=66,
            font=FONT, wrap=tk.WORD,
            bg=BG_QBOX, fg=FG_WHITE,
            insertbackground=FG_WHITE,
            relief=tk.FLAT, bd=4
        )
        self.question_box.pack(padx=16, pady=(2, 8), fill=tk.X)
        self._load_equation(self.planner.current_equation())

        self.hint_var = tk.StringVar(value="")
        tk.Label(
            root, textvariable=self.hint_var,
            font=FONT_BOLD, bg=BG_MAIN, fg="#1a7a1a",
            wraplength=660, justify=tk.LEFT
        ).pack(anchor=tk.W, padx=16, pady=(0, 6))

        tk.Label(root, text="Enter your solution steps (one per line):",
                 font=FONT_BOLD, bg=BG_MAIN, fg="#1a1a5e").pack(anchor=tk.W, padx=14)
        self.steps_input = tk.Text(
            root, height=6, width=66,
            font=("Comic Sans MS", 14), bg="#ffffff", fg="#000000",
            insertbackground="#000000", insertontime=600, insertofftime=300,
            relief=tk.GROOVE, bd=2
        )
        self.steps_input.pack(padx=16, pady=(2, 4), fill=tk.X)

        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            root, textvariable=self.progress_var,
            font=FONT, bg=BG_MAIN, fg="#007a00"
        )
        self.progress_label.pack()

        self.steps_input.bind("<Return>", self.on_step_entered)

        btn_frame = tk.Frame(root, bg=BG_MAIN)
        btn_frame.pack(pady=10)
        _make_btn(btn_frame, "Done",       self.solve).pack(side=tk.LEFT, padx=6)
        self.btn_next = _make_btn(btn_frame, "Next", self.next_equation)
        self.btn_next.config(bg=BTN_DIS)
        self.btn_next.pack(side=tk.LEFT, padx=6)
        _make_btn(btn_frame, "Start Over", self.reset_questions).pack(side=tk.LEFT, padx=6)

        _make_btn(root, "Hint", self.astar_hint).pack(pady=5)

        tk.Label(root, text="Debug Details",
                 font=FONT_BOLD, bg=BG_MAIN, fg="#1a1a5e").pack(anchor=tk.W, padx=14, pady=(8, 0))
        self.output = tk.Text(
            root, height=15, width=66,
            font=FONT, bg=BG_OUTPUT, fg=FG_BLACK,
            relief=tk.GROOVE, bd=2
        )
        self.output.pack(padx=16, pady=(4, 16), fill=tk.X)
        self.output.tag_configure("correct",  foreground="#007a00", font=("Comic Sans MS", 14, "bold"))
        self.output.tag_configure("incorrect", foreground="#cc0000", font=("Comic Sans MS", 14, "bold"))
        self.output.tag_configure("heading",   foreground="#1a1a5e", font=("Comic Sans MS", 13, "bold"))

    def _extract_equation(self, raw: str) -> str:
        import re as _re
        m = _re.search(r"\[equation:\s*(.+?)\]", raw)
        if m:
            return m.group(1).strip()
        return raw.split("\n")[0].strip()

    def _load_equation(self, raw: str):
        eq = self._extract_equation(raw)
        self._current_equation = eq

        if "[equation:" in raw:
            sentence = raw.split("\n[equation:")[0].strip()
            display = sentence
            self._is_word_problem = True
        else:
            display = eq
            self._is_word_problem = False
        self._wp_equation_hint_shown = False
        if hasattr(self, "hint_var"):
            self.hint_var.set("")

        self.question_box.config(state=tk.NORMAL)
        self.question_box.delete("1.0", tk.END)
        self.question_box.insert(tk.END, display)
        self.question_box.config(state=tk.DISABLED)

    def _lock_steps(self):
        self.steps_input.config(state=tk.DISABLED)

    def _unlock_steps(self):
        self.steps_input.config(state=tk.NORMAL)

    def _update_score(self):
        self.score_var.set(f"Score: {self._correct_count} / {self._total_count}")

    def next_equation(self):
        if self._last_answer_correct:
            raw = self.planner.next_equation()
        else:
            raw = self.planner.repeat_level()
        self._last_answer_correct = True
        self._question_num += 1
        self.question_label_var.set(f"{self._question_num}. Question")
        self._load_equation(raw)
        self._unlock_steps()
        self.steps_input.delete("1.0", tk.END)
        self.output.delete("1.0", tk.END)
        self.progress_var.set("")
        self.progress_label.config(fg="#007a00")
        self.btn_next.config(bg="#7a8a9e")

    def reset_questions(self):
        self.planner.reset()
        self._question_num = 1
        self._correct_count = 0
        self._total_count = 0
        self._update_score()
        self.question_label_var.set("1. Question")
        raw = self.planner.current_equation()
        self._load_equation(raw)
        self._unlock_steps()
        self.steps_input.delete("1.0", tk.END)
        self.output.delete("1.0", tk.END)
        self.progress_var.set("")
        self.progress_label.config(fg="#007a00")
        self.btn_next.config(bg="#7a8a9e")

    def astar_hint(self):
        equation = self._current_equation.strip()
        if not equation or "=" not in equation:
            self.hint_var.set("Please enter a valid equation first.")
            return

        if getattr(self, "_is_word_problem", False) and not self._wp_equation_hint_shown:
            self.hint_var.set(f"Hint 1 — Equation: {equation}")
            self._wp_equation_hint_shown = True
            return

        raw_steps = self.steps_input.get("1.0", tk.END).strip()
        start_equation = equation
        if raw_steps:
            for line in reversed(raw_steps.split("\n")):
                line = line.strip()
                if not line:
                    continue
                if "=" in line and line.count("=") == 1:
                    start_equation = line
                    break

        try:
            import re as _re
            def _is_quadratic_str(s):
                ns = s.replace(" ", "")
                if "x**2" in ns or "x^2" in ns:
                    return True
                if _re.search(r'x\s*\(', ns):
                    return True
                if _re.search(r'\([^)]+\)\s*\*?\s*\([^)]+\)', ns):
                    return True
                return False
            is_quadratic = _is_quadratic_str(start_equation) or _is_quadratic_str(equation)
            if is_quadratic:
                hint = astar_next_step_factor_quadratic(start_equation)
                if not hint:
                    hint = astar_next_step_complete_square(start_equation)
            else:
                hint = astar_next_step_isolate(start_equation, target_var="x")
        except Exception:
            hint = None

        if not hint:
            self.hint_var.set("A* Hint: No suggestion available.")
            return

        self.hint_var.set(f"A* Hint: {hint}")

    def on_step_entered(self, event):
        self.root.after(1, self.update_progress)
        return None

    def update_progress(self):
        equation = self._current_equation.strip()
        if not equation or "=" not in equation:
            self.progress_var.set("")
            return

        raw = self.steps_input.get("1.0", tk.END).strip()
        if not raw:
            self.progress_var.set("")
            return

        student_steps = raw.split("\n")
        error_index, result = validate_steps(student_steps, equation)

        if error_index >= 0:
            self.progress_label.config(fg="#cc0000")
            self.progress_var.set(f"✘  Step {error_index+1}: {result}")
            return

        self.progress_label.config(fg="#007a00")
        is_word_problem = getattr(self, "_is_word_problem", False)

        def _acceptable_solution_format(s):
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

        def _last_solution_rhs_text(steps):
            for line in reversed(steps):
                t = line.strip()
                if not t:
                    continue
                clauses = [c.strip() for c in re.split(r"\bor\b", t, flags=re.IGNORECASE) if c.strip()]
                for clause in clauses:
                    if clause.count("=") != 1:
                        continue
                    left, right = [p.strip() for p in clause.split("=")]
                    if left.replace(" ", "") == "x":
                        return right
                    if right.replace(" ", "") == "x":
                        return left
            return None

        wp_bonus = 1 if is_word_problem else 0
        eq_step_done = _student_has_equation_step(student_steps, equation) if is_word_problem else False

        rhs_text = _last_solution_rhs_text(student_steps)
        has_final_answer = rhs_text is not None and _acceptable_solution_format(rhs_text)

        if error_index == -2 or error_index == -1:
            if is_word_problem:
                if has_final_answer:
                    self.progress_var.set("100% complete")
                elif eq_step_done:
                    self.progress_var.set("50% complete — now write the final answer")
                else:
                    self.progress_var.set("In progress")
            elif error_index == -1:
                if has_final_answer:
                    self.progress_var.set("100% complete")
                else:
                    self.progress_var.set("Almost complete: format final answer as a decimal (>= 2 dp) or a proper fraction")
            else:
                capped = re.sub(r"(\d+)%", lambda m: f"{min(int(m.group(1)), 99)}%", result)
                self.progress_var.set(capped)
        else:
            self.progress_var.set("")

    def solve(self):
        self.btn_next.config(bg="#1a3a6e")
        self._lock_steps()
        self._total_count += 1
        equation = self._current_equation
        student_steps = self.steps_input.get("1.0", tk.END).strip().split("\n")

        self.output.delete(1.0, tk.END)

        try:
            steps, solution = solve_equation_steps(equation)

            self.output.insert(tk.END, "=== Expected Solution ===\n")
            for step in steps:
                self.output.insert(tk.END, step + "\n")

            has_steps = any(s.strip() for s in student_steps)

            if not has_steps:
                self.progress_label.config(fg="#cc0000")
                self.progress_var.set("✘  No steps entered")
                self.student.update(False)
                self._last_answer_correct = False
                self.output.insert(tk.END, "\nNo steps entered.\n", "incorrect")
                self.output.insert(tk.END, "\nHint: Enter your working steps above and click Done to check them.\n")
            else:
                error_index, result = validate_steps(student_steps, equation)

                if error_index == -1:
                    self._correct_count += 1
                    self.student.update(True)
                    self._last_answer_correct = True
                    self.progress_label.config(fg="#007a00")
                    self.progress_var.set("✔  All steps correct!")
                    self.output.insert(tk.END, "\n✔  All steps correct!\n")
                    self.output.insert(tk.END, "Hint: Well done! Click Next for the next question.\n")
                elif error_index == -2:
                    self.student.update(False)
                    self._last_answer_correct = False
                    clean = result.replace("Valid so far: ", "").replace("about ", "")
                    self.progress_label.config(fg="#cc0000")
                    self.progress_var.set(f"◑  Incomplete — {clean}")
                    self.output.insert(tk.END, f"\n◑  Incomplete — {clean}\n", "incorrect")
                    self.output.insert(tk.END, "Hint: Keep going — finish solving for x.\n")
                else:
                    self.student.update(False)
                    self._last_answer_correct = False
                    self.progress_label.config(fg="#cc0000")
                    self.progress_var.set(f"✘  Step {error_index+1}: {result}")
                    self.output.insert(tk.END, f"\n✘  Error at Step {error_index+1}: {result}\n", "incorrect")

            self._update_score()
            self.output.insert(tk.END, f"\nDifficulty: {self.student.difficulty}\n")

        except Exception as e:
            self.output.insert(tk.END, f"\nError: {e}\n")

def run_app():
    root = tk.Tk()
    app = MathTutorApp(root)
    root.mainloop()
