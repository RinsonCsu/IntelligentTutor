import tkinter as tk
from tkinter import messagebox
import re
from solver import (
    solve_equation_steps,
    astar_next_step_isolate,
    astar_next_step_complete_square,
    astar_next_step_factor_quadratic,
)
from validator import validate_steps, estimate_linear_steps_remaining
from model import classify_error, error_labels
from hints import generate_hint
from adaptive import StudentModel, generate_problem
from symbolic_planner import SymbolicPlanner

class MathTutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Math Teacher")

        FONT       = ("Comic Sans MS", 14)
        FONT_BOLD  = ("Comic Sans MS", 14, "bold")
        FONT_TITLE = ("Comic Sans MS", 16, "bold")
        BG_MAIN    = "#1a1a5e"
        BG_QBOX    = "#0d0d3b"
        BG_STEPS   = "#f0f8ff"
        BG_OUTPUT  = "#ffffff"
        FG_WHITE   = "#ffffff"
        FG_BLACK   = "#000000"
        FG_YELLOW  = "#ffd700"
        BTN_SOLVE  = {"bg": "#28a745", "fg": FG_BLACK, "font": FONT_BOLD, "relief": tk.RAISED, "bd": 3}
        BTN_NEXT   = {"bg": "#007bff", "fg": FG_BLACK, "font": FONT_BOLD, "relief": tk.RAISED, "bd": 3}
        BTN_RESET  = {"bg": "#cc0000", "fg": FG_BLACK, "font": FONT_BOLD, "relief": tk.RAISED, "bd": 3}
        BTN_HINT   = {"bg": "#ff8c00", "fg": FG_BLACK, "font": FONT_BOLD, "relief": tk.RAISED, "bd": 3}

        root.configure(bg=BG_MAIN)
        root.minsize(720, 700)

        self.student = StudentModel()
        self.planner = SymbolicPlanner()
        self._question_num = 1
        self._current_equation = ""
        self._wp_equation_hint_shown = False

        tk.Label(root, text="Intelligent Math Teacher",
                 font=("Comic Sans MS", 20, "bold"),
                 bg=BG_MAIN, fg=FG_YELLOW).pack(pady=(14, 6))

        self.question_label_var = tk.StringVar(value="1. Question")
        tk.Label(root, textvariable=self.question_label_var,
                 font=FONT_BOLD, bg=BG_MAIN, fg=FG_YELLOW).pack(anchor=tk.W, padx=14, pady=(6, 0))

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
            font=FONT_BOLD, bg=BG_MAIN, fg=FG_YELLOW,
            wraplength=660, justify=tk.LEFT
        ).pack(anchor=tk.W, padx=16, pady=(0, 6))

        tk.Label(root, text="Enter your solution steps (one per line):",
                 font=FONT, bg=BG_MAIN, fg=FG_WHITE).pack()
        self.steps_input = tk.Text(
            root, height=6, width=66,
            font=FONT, bg=BG_STEPS, fg=FG_BLACK,
            insertbackground=FG_BLACK, insertontime=600, insertofftime=300,
            relief=tk.GROOVE, bd=2
        )
        self.steps_input.pack(padx=16, pady=(2, 4), fill=tk.X)

        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            root, textvariable=self.progress_var,
            font=FONT, bg=BG_MAIN, fg="#90ee90"
        )
        self.progress_label.pack()

        self.steps_input.bind("<Return>", self.on_step_entered)

        btn_frame = tk.Frame(root, bg=BG_MAIN)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Solve",  command=self.solve,           **BTN_SOLVE).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Next",   command=self.next_equation,   **BTN_NEXT ).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Reset",  command=self.reset_questions, **BTN_RESET).pack(side=tk.LEFT, padx=6)

        tk.Button(root, text="A* Hint", command=self.astar_hint, **BTN_HINT).pack(pady=5)

        self.output = tk.Text(
            root, height=15, width=66,
            font=FONT, bg=BG_OUTPUT, fg=FG_BLACK,
            relief=tk.GROOVE, bd=2
        )
        self.output.pack(padx=16, pady=(4, 16), fill=tk.X)

    def _extract_equation(self, raw: str) -> str:
        """Extract solver-compatible equation from a word problem or plain equation string."""
        import re as _re
        m = _re.search(r"\[equation:\s*(.+?)\]", raw)
        if m:
            return m.group(1).strip()
        return raw.split("\n")[0].strip()

    def _load_equation(self, raw: str):
        """Load a new equation (plain or word problem) into the question box."""
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

    def next_equation(self):
        raw = self.planner.next_equation()
        self._question_num += 1
        self.question_label_var.set(f"{self._question_num}. Question")
        self._load_equation(raw)
        self.steps_input.delete("1.0", tk.END)
        self.output.delete("1.0", tk.END)
        self.progress_var.set("")

    def reset_questions(self):
        self.planner.reset()
        self._question_num = 1
        self.question_label_var.set("1. Question")
        raw = self.planner.current_equation()
        self._load_equation(raw)
        self.steps_input.delete("1.0", tk.END)
        self.output.delete("1.0", tk.END)
        self.progress_var.set("")

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
            start_no_space = start_equation.replace(" ", "")
            is_quadratic = ("x**2" in start_no_space) or ("x^2" in start_no_space)
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

        if error_index == -2:
            self.progress_var.set(result)
        elif error_index == -1:
            rhs_text = _last_solution_rhs_text(student_steps)
            if rhs_text is not None and _acceptable_solution_format(rhs_text):
                self.progress_var.set("100% complete")
            else:
                start_steps, cur_steps = estimate_linear_steps_remaining(equation, f"x={rhs_text}" if rhs_text else "x=0")
                if start_steps is None or cur_steps is None:
                    self.progress_var.set("Almost complete: format final answer as a decimal (>= 2 dp) or a proper fraction")
                else:
                    total_steps = start_steps + 1
                    remaining_steps = cur_steps + 1
                    pct = int(100 * (total_steps - remaining_steps) / max(1, total_steps))
                    pct = max(0, min(99, pct))
                    self.progress_var.set(
                        f"{pct}% complete: format final answer as a decimal (>= 2 dp) or a proper fraction"
                    )
        else:
            self.progress_var.set("")

    def solve(self):
        equation = self._current_equation
        student_steps = self.steps_input.get("1.0", tk.END).strip().split("\n")

        self.output.delete(1.0, tk.END)

        try:
            steps, solution = solve_equation_steps(equation)

            self.output.insert(tk.END, "=== Expected Solution ===\n")
            for step in steps:
                self.output.insert(tk.END, step + "\n")

            error_index, result = validate_steps(student_steps, equation)

            if error_index == -1:
                self.output.insert(tk.END, "\nAll steps correct!\n")
                self.student.update(True)
                error_code = 0
                predicted = 0
                feedback_hint = generate_hint(predicted)
                feedback_label = error_labels[predicted]
            elif error_index == -2:
                self.output.insert(tk.END, f"\n{result}\n")
                self.student.update(False)
                feedback_label = "In Progress"
                feedback_hint = result
            else:
                self.output.insert(tk.END, f"Error at Step {error_index+1}: {result}\n")
                self.student.update(False)
                error_code = 3
                predicted = classify_error(error_code)
                feedback_label = error_labels[predicted]
                feedback_hint = generate_hint(predicted)

            self.output.insert(tk.END, "\n=== Feedback ===\n")
            self.output.insert(tk.END, f"Detected: {feedback_label}\n")
            self.output.insert(tk.END, f"Hint: {feedback_hint}\n")

            self.output.insert(tk.END, f"\nDifficulty: {self.student.difficulty}\n")

        except Exception as e:
            self.output.insert(tk.END, f"\nError: {e}\n")

def run_app():
    root = tk.Tk()
    app = MathTutorApp(root)
    root.mainloop()
