import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    convert_xor, implicit_multiplication_application,
)

error_labels = {
    0: "Correct",
    1: "Arithmetic Error",
    2: "Sign Error",
    3: "Algebraic Error",
    4: "Incomplete",
    5: "Invalid"
}

_NUM_CLASSES = len(error_labels)
_INPUT_DIM   = 6
_TF = standard_transformations + (convert_xor, implicit_multiplication_application,)
_sym_x = sp.Symbol("x")


class _ErrorClassifier(nn.Module):
    def __init__(self, input_dim=_INPUT_DIM, hidden_dim=64, num_classes=_NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.fc3  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3  = nn.LayerNorm(hidden_dim // 2)
        self.drop = nn.Dropout(p=dropout)
        self.out  = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.drop(F.relu(self.ln1(self.fc1(x))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)


def _parse(s):
    try:
        return parse_expr(s, transformations=_TF)
    except Exception:
        return None


def extract_features(prev_step: str, curr_step: str, equation: str) -> torch.Tensor:
    feats = [0.0] * _INPUT_DIM
    try:
        if "=" not in prev_step or "=" not in curr_step:
            feats[5] = 1.0
            return torch.tensor([feats], dtype=torch.float32)

        pl, pr = prev_step.split("=", 1)
        cl, cr = curr_step.split("=", 1)
        p_lhs, p_rhs = _parse(pl), _parse(pr)
        c_lhs, c_rhs = _parse(cl), _parse(cr)

        if any(v is None for v in (p_lhs, p_rhs, c_lhs, c_rhs)):
            feats[5] = 1.0
            return torch.tensor([feats], dtype=torch.float32)

        prev_expr = sp.simplify(p_lhs - p_rhs)
        curr_expr = sp.simplify(c_lhs - c_rhs)

        try:
            orig_l, orig_r = equation.split("=", 1)
            orig_eq = sp.Eq(_parse(orig_l), _parse(orig_r))
            sols = sp.solve(sp.Eq(c_lhs, c_rhs), _sym_x)
            if sols:
                res = float(abs(sp.simplify(
                    orig_eq.lhs.subs(_sym_x, sols[0]) - orig_eq.rhs.subs(_sym_x, sols[0])
                )))
                feats[0] = min(res, 50.0) / 50.0
        except Exception:
            pass

        try:
            feats[1] = 1.0 if sp.simplify(prev_expr + curr_expr) == 0 else 0.0
        except Exception:
            pass

        try:
            lhs_same = sp.simplify(p_lhs - c_lhs) == 0
            rhs_same = sp.simplify(p_rhs - c_rhs) == 0
            feats[2] = 1.0 if (lhs_same != rhs_same) else 0.0
        except Exception:
            pass

        try:
            equivalent = sp.simplify(prev_expr - curr_expr) == 0
            if not equivalent and curr_expr != 0:
                ratio = sp.simplify(prev_expr / curr_expr)
                equivalent = (getattr(ratio, "free_symbols", set()) == set()
                              and sp.simplify(prev_expr - ratio * curr_expr) == 0)
            feats[3] = 0.0 if equivalent else 1.0
        except Exception:
            feats[3] = 1.0

        try:
            from expert_system import count_steps_remaining
            rem = count_steps_remaining(curr_step)
            feats[4] = min(rem or 0, 5) / 5.0
        except Exception:
            pass

    except Exception:
        feats[5] = 1.0

    return torch.tensor([feats], dtype=torch.float32)


def _make_training_data():
    data = [
        ([0.0, 0.0, 0.0, 0.0, 0.2, 0.0], 0),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        ([0.4, 0.0, 0.0, 1.0, 0.2, 0.0], 1),
        ([0.6, 0.0, 0.0, 1.0, 0.0, 0.0], 1),
        ([0.0, 1.0, 0.0, 1.0, 0.2, 0.0], 2),
        ([0.0, 1.0, 0.0, 1.0, 0.4, 0.0], 2),
        ([0.0, 0.0, 1.0, 1.0, 0.2, 0.0], 3),
        ([0.2, 0.0, 1.0, 1.0, 0.4, 0.0], 3),
        ([0.0, 0.0, 0.0, 0.0, 0.8, 0.0], 4),
        ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 4),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 5),
        ([0.0, 0.0, 0.0, 1.0, 0.0, 1.0], 5),
    ]
    xs = torch.tensor([d[0] for d in data], dtype=torch.float32)
    ys = torch.tensor([d[1] for d in data], dtype=torch.long)
    return xs, ys


def _train(model, epochs=300):
    xs, ys = _make_training_data()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(xs), ys)
        loss.backward()
        opt.step()
    model.eval()


import sys as _sys
_BASE_DIR   = getattr(_sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "error_classifier.pt")

torch.manual_seed(0)
_model = _ErrorClassifier()

if os.path.isfile(_MODEL_PATH):
    _model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
    _model.eval()
    print(f"[model] Loaded error classifier from {_MODEL_PATH}")
else:
    print("[model] Training error classifier...")
    _train(_model)
    torch.save(_model.state_dict(), _MODEL_PATH)
    print(f"[model] Saved error classifier to {_MODEL_PATH}")
    print(f"[model] Delete that file to force retraining.")


def classify_error_from_steps(prev_step: str, curr_step: str, equation: str) -> tuple:
    feats = extract_features(prev_step, curr_step, equation)
    with torch.no_grad():
        logits = _model(feats)
        probs  = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    confidence = float(probs[idx])
    return idx, error_labels[idx], confidence
