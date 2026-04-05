import random

class StudentModel:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.difficulty = "Easy"

    def update(self, correct):
        if correct:
            self.correct += 1
        else:
            self.incorrect += 1
        self.adjust()

    def adjust(self):
        total = self.correct + self.incorrect
        if total < 3:
            self.difficulty = "Easy"
        elif self.correct / max(1, total) > 0.7:
            self.difficulty = "Hard"
        else:
            self.difficulty = "Medium"

def generate_problem(level):
    if level == "Easy":
        return "2*x + 3 = 7"
    elif level == "Medium":
        return "3*x - 5 = 10"
    else:
        return "2*x + 3 = x + 9"
