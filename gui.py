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

class MathTutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Math Teacher")

        self.student = StudentModel()

        tk.Label(root, text="Enter Equation (e.g., 2*x+3=7):").pack()
        self.eq_entry = tk.Entry(root, width=40)
        self.eq_entry.pack()

        tk.Label(root, text="Enter your solution steps (one per line):").pack()
        self.steps_input = tk.Text(root, height=6, width=50)
        self.steps_input.pack()

        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(root, textvariable=self.progress_var)
        self.progress_label.pack()

        self.steps_input.bind("<Return>", self.on_step_entered)

        tk.Button(root, text="Solve", command=self.solve).pack(pady=10)

        tk.Button(root, text="A* Hint", command=self.astar_hint).pack(pady=5)

        self.output = tk.Text(root, height=15, width=60)
        self.output.pack()

    def astar_hint(self):
        equation = self.eq_entry.get().strip()
        if not equation or "=" not in equation:
            messagebox.showerror("Error", "Invalid input.")
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
            self.output.insert(tk.END, "\nA* Hint: No suggestion available.\n")
            return

        self.output.insert(tk.END, f"\nA* Hint (next step): {hint}\n")

    def on_step_entered(self, event):
        self.root.after(1, self.update_progress)
        return None

    def update_progress(self):
        equation = self.eq_entry.get().strip()
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
        equation = self.eq_entry.get()
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
            self.output.insert(tk.END, f"Next Problem: {generate_problem(self.student.difficulty)}\n")

        except:
            messagebox.showerror("Error", "Invalid input.")

def run_app():
    root = tk.Tk()
    app = MathTutorApp(root)
    root.mainloop()
