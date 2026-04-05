import torch
import torch.nn as nn

error_labels = {
    0: "Correct",
    1: "Arithmetic Error",
    2: "Sign Error",
    3: "Algebraic Error",
    4: "Incomplete",
    5: "Invalid"
}

_NUM_CLASSES = len(error_labels)


class _ErrorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, _NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
_model = _ErrorClassifier()
_model.eval()


def classify_error(code: int) -> int:
    """Map an integer error code to an error label index using the PyTorch classifier."""
    x = torch.tensor([[float(code)]], dtype=torch.float32)
    with torch.no_grad():
        logits = _model(x)
    return int(torch.argmax(logits, dim=1).item())
