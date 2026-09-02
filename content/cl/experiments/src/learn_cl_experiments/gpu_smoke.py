"""Optional GPU smoke: sequential LoRA-like updates on random tensors.

No Hugging Face download. If torch is missing, print skip and exit 0 from CLI.
"""

from __future__ import annotations

from typing import Any


def _run_torch() -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    dim = 32
    rank = 4
    base = torch.nn.Linear(dim, dim, bias=False).to(device)
    with torch.no_grad():
        base.weight.copy_(torch.eye(dim, device=device))

    def lora_update(shift: torch.Tensor, steps: int) -> None:
        a = torch.nn.Parameter(torch.zeros(rank, dim, device=device))
        b = torch.nn.Parameter(0.01 * torch.randn(dim, rank, device=device))
        opt = torch.optim.SGD([a, b], lr=0.2)
        x = torch.eye(dim, device=device)
        target = x + shift
        for _ in range(steps):
            opt.zero_grad()
            delta = b @ a
            pred = torch.nn.functional.linear(x, base.weight + delta)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            opt.step()
        with torch.no_grad():
            base.weight.add_(b @ a)

    task_a = 0.4 * torch.eye(dim, device=device)
    task_b = -0.4 * torch.eye(dim, device=device)

    def score(shift: torch.Tensor) -> float:
        x = torch.eye(dim, device=device)
        pred = torch.nn.functional.linear(x, base.weight)
        return float(torch.nn.functional.mse_loss(pred, x + shift).item())

    before_a = score(task_a)
    lora_update(task_a, steps=40)
    after_a = score(task_a)
    after_a_then_b_on_a: float
    lora_update(task_b, steps=40)
    after_b = score(task_b)
    after_a_then_b_on_a = score(task_a)

    return {
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "task_a_before": before_a,
        "task_a_after_a": after_a,
        "task_b_after_b": after_b,
        "task_a_after_b": after_a_then_b_on_a,
        "forgot_a": after_a_then_b_on_a > after_a + 0.01,
        "learned_a": after_a < before_a - 0.01,
    }


def run_smoke() -> dict[str, Any]:
    try:
        return _run_torch()
    except ImportError:
        return {
            "device": "none",
            "cuda": False,
            "skipped": True,
            "reason": "torch 未安装。CPU 机制实验不需要它；GPU 教程见 GPU.md。",
        }
