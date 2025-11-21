import sys
import os
import subprocess
from typing import List, Tuple

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from third_party.tasks.associativetask import dataloader
from snntorch._neurons.stateleaky import StateLeaky


def get_least_busy_gpu() -> int:
    try:
        result: str = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        memory_used: List[int] = [int(x) for x in result.strip().split("\n")]
        if memory_used:
            return memory_used.index(min(memory_used))
        return 0
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi failed or isn't available, defaulting to GPU 0")
        return 0


DEVICE = f"cuda:{get_least_busy_gpu()}" if torch.cuda.is_available() else "cpu"


class StateLeakyAssocModel(nn.Module):
    def __init__(self, seq_width: int, hidden_dim: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(seq_width + 1, hidden_dim)
        self.lif1 = StateLeaky(beta=0.9, channels=hidden_dim)
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lif2 = StateLeaky(beta=0.9, channels=hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, seq_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, W+1)
        T, B, _ = x.shape
        h, _ = self.lif1(self.in_proj(x))  # (T,B,H)
        h = h.reshape(-1, h.shape[-1])
        h = self.mid(h)
        h = h.reshape(T, B, -1)
        h, _ = self.lif2(h)
        h = h.reshape(-1, h.shape[-1])
        y = self.out_proj(h).reshape(T, B, -1)
        return torch.sigmoid(y)


def evaluate_stateleaky_assoc(
    num_batches: int = 200,
    batch_size: int = 32,
    seq_width: int = 8,
    num_pairs_min: int = 2,
    num_pairs_max: int = 6,
    hidden_dim: int = 256,
) -> Tuple[float, float]:
    model = StateLeakyAssocModel(
        seq_width=seq_width, hidden_dim=hidden_dim
    ).to(DEVICE)
    model.eval()
    bce_sum = 0.0
    total_bits = 0
    total_correct = 0
    bce = nn.BCELoss(reduction="sum")

    with torch.no_grad():
        for _, inp, target in tqdm(
            dataloader(
                num_batches,
                batch_size,
                seq_width,
                num_pairs_min,
                num_pairs_max,
            ),
            total=num_batches,
            desc="StateLeaky Assoc eval",
        ):
            inp = inp.to(DEVICE)  # (2N+2, B, W+1)
            target = target.to(DEVICE)  # (1, B, W)
            y = model(inp)  # (2N+2, B, W)
            y_last = y[-1:]  # (1, B, W)
            loss = bce(y_last, target)
            bce_sum += float(loss.item())
            pred = (y_last >= 0.5).float()
            total_bits += int(target.numel())
            total_correct += int((pred == target).sum().item())

    avg_loss = bce_sum / max(1, num_batches * batch_size)
    bit_acc = total_correct / max(1, total_bits)
    return avg_loss, bit_acc


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    # Hyperparameters
    total_steps = 50000
    eval_every = 500
    log_every = 50
    batch_size = 32
    seq_width = 8
    num_pairs_min = 2
    num_pairs_max = 6
    hidden_dim = 256
    lr = 3e-4
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), "checkpoints_assoc_stateleaky"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(
        project="snntorch-associative-recall",
        name=run_name,
        config={
            "model": "StateLeaky",
            "device": DEVICE,
            "total_steps": total_steps,
            "eval_every": eval_every,
            "log_every": log_every,
            "batch_size": batch_size,
            "seq_width": seq_width,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "num_pairs_min": num_pairs_min,
            "num_pairs_max": num_pairs_max,
        },
    )

    model = StateLeakyAssocModel(
        seq_width=seq_width, hidden_dim=hidden_dim
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="mean")

    # Fixed eval set
    def build_eval_set(num_batches: int = 50):
        state_rand = __import__("random").getstate()
        state_np = __import__("numpy").random.get_state()
        __import__("random").seed(8888)
        __import__("numpy").random.seed(8888)
        data = list(
            dataloader(
                num_batches,
                batch_size,
                seq_width,
                num_pairs_min,
                num_pairs_max,
            )
        )
        __import__("random").setstate(state_rand)
        __import__("numpy").random.set_state(state_np)
        return data

    eval_batches = build_eval_set(50)
    best_loss = float("inf")

    # Training stream
    train_iter = dataloader(
        total_steps, batch_size, seq_width, num_pairs_min, num_pairs_max
    )

    model.train()
    for step, (batch_idx, inp, target) in enumerate(
        tqdm(train_iter, total=total_steps, desc="StateLeaky Assoc train"),
        start=1,
    ):
        inp = inp.to(DEVICE)  # (2N+2,B,W+1)
        target = target.to(DEVICE)  # (1,B,W)

        optimizer.zero_grad(set_to_none=True)
        y = model(inp)
        y_last = y[-1:]
        loss = criterion(y_last, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_every == 0:
            with torch.no_grad():
                pred = (y_last >= 0.5).float()
                bit_acc = (pred == target).float().mean().item()
            wandb.log(
                {
                    "train_loss": float(loss.item()),
                    "train_bit_acc": bit_acc,
                    "step": step,
                }
            )

        if step % eval_every == 0:
            model.eval()
            bce_sum = 0.0
            total_bits = 0
            total_correct = 0
            with torch.no_grad():
                for _, inp_e, target_e in eval_batches:
                    inp_e = inp_e.to(DEVICE)
                    target_e = target_e.to(DEVICE)
                    out = model(inp_e)
                    out_last = out[-1:]
                    l = nn.functional.binary_cross_entropy(
                        out_last, target_e, reduction="sum"
                    )
                    bce_sum += float(l.item())
                    pred_e = (out_last >= 0.5).float()
                    total_bits += int(target_e.numel())
                    total_correct += int((pred_e == target_e).sum().item())
            avg_loss = bce_sum / max(1, len(eval_batches) * batch_size)
            bit_accuracy = total_correct / max(1, total_bits)
            wandb.log(
                {
                    "eval_loss": avg_loss,
                    "eval_bit_acc": bit_accuracy,
                    "step": step,
                }
            )
            # checkpoints
            last_path = os.path.join(ckpt_dir, "stateleaky_last.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                last_path,
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(ckpt_dir, "stateleaky_best.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                    best_path,
                )
            model.train()

        if step >= total_steps:
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
