import sys
import os
import subprocess
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from third_party.tasks.copytask import dataloader
from snntorch._neurons.associative import AssociativeMemorySSM


def get_least_busy_gpu() -> int:
    """Return the index of the GPU with the least memory usage."""
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


class Gen2CopyModel(nn.Module):
    def __init__(self, seq_width: int, hidden_dim: int = 256):
        super().__init__()
        # Gen2 requires square num_spiking_neurons
        assert (
            int(hidden_dim**0.5) ** 2 == hidden_dim
        ), "hidden_dim must be a perfect square for Gen2"
        self.in_proj = nn.Linear(seq_width + 1, hidden_dim)
        self.gen2_1 = AssociativeMemorySSM.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            use_q_projection=True,
        )
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gen2_2 = AssociativeMemorySSM.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            use_q_projection=True,
        )
        self.out_proj = nn.Linear(hidden_dim, seq_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, seq_width+1)
        T, B, _ = x.shape
        h = self.in_proj(x)  # (T, B, H)
        h = self.gen2_1(h)  # spikes (T, B, H)
        h = h.reshape(-1, h.shape[-1])
        h = self.mid(h)  # (T*B, H)
        h = h.reshape(T, B, -1)
        h = self.gen2_2(h)  # (T, B, H)
        h = h.reshape(-1, h.shape[-1])
        y = self.out_proj(h)  # (T*B, seq_width)
        y = y.reshape(T, B, -1)
        return torch.sigmoid(y)


def evaluate_gen2(
    num_batches: int = 200,
    batch_size: int = 32,
    seq_width: int = 8,
    min_len: int = 1,
    max_len: int = 20,
    hidden_dim: int = 256,
) -> Tuple[float, float]:
    model = Gen2CopyModel(seq_width=seq_width, hidden_dim=hidden_dim).to(
        DEVICE
    )
    model.eval()
    total_loss = 0.0
    total_bits = 0
    total_correct = 0
    bce = nn.BCELoss(reduction="sum")

    with torch.no_grad():
        for _, inp, target in dataloader(
            num_batches, batch_size, seq_width, min_len, max_len
        ):
            inp = inp.to(DEVICE)  # (T+1, B, seq_width+1)
            target = target.to(DEVICE)  # (T, B, seq_width)
            # Model expects (T, B, C) aligned to target T; drop the delimiter step
            x = inp[:-1]
            logits = model(x)  # (T, B, seq_width), sigmoid outputs

            loss = bce(logits, target)
            total_loss += float(loss.item())

            pred = (logits >= 0.5).float()
            total_bits += int(target.numel())
            total_correct += int((pred == target).sum().item())

    avg_loss = total_loss / max(1, num_batches * batch_size)
    bit_acc = total_correct / max(1, total_bits)
    return avg_loss, bit_acc


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    # Hyperparameters (simple, task-appropriate)
    total_steps = 50000
    eval_every = 500
    log_every = 50
    batch_size = 32
    seq_width = 8
    min_len = 1
    max_len = 20
    hidden_dim = 256
    lr = 1e-3
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), "checkpoints_copytask_gen2"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(
        project="snntorch-ntm-copy",
        name=run_name,
        config={
            "model": "Gen2",
            "device": DEVICE,
            "total_steps": total_steps,
            "eval_every": eval_every,
            "log_every": log_every,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "min_len": min_len,
            "max_len": max_len,
        },
    )

    # Model/optimizer/loss
    model = Gen2CopyModel(seq_width=seq_width, hidden_dim=hidden_dim).to(
        DEVICE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="mean")

    # Build a small fixed eval set (same length range, fixed seed)
    def build_eval_set(num_batches: int = 50):
        state_rand = __import__("random").getstate()
        state_np = __import__("numpy").random.get_state()
        __import__("random").seed(1234)
        __import__("numpy").random.seed(1234)
        data = list(
            dataloader(num_batches, batch_size, seq_width, min_len, max_len)
        )
        __import__("random").setstate(state_rand)
        __import__("numpy").random.set_state(state_np)
        return data

    eval_batches = build_eval_set(50)

    best_loss = float("inf")
    step = 0

    # Training generator
    train_iter = dataloader(
        total_steps, batch_size, seq_width, min_len, max_len
    )

    model.train()
    for batch_idx, inp, target in tqdm(
        train_iter, total=total_steps, desc="Gen2 train"
    ):
        step = batch_idx
        inp = inp.to(DEVICE)  # (L+1, B, seq_width+1) including delimiter
        target = target.to(DEVICE)  # (L, B, seq_width)
        L = target.shape[0]
        B = target.shape[1]
        C = inp.shape[2]
        blanks = torch.zeros(L, B, C, device=DEVICE)
        x = torch.cat([inp, blanks], dim=0)  # (2L+1, B, C)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # (2L+1,B,seq_width), sigmoid
        y_write = logits[-L:]  # last L steps correspond to the write window
        loss = criterion(y_write, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_every == 0:
            with torch.no_grad():
                pred = (y_write >= 0.5).float()
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
            total_loss = 0.0
            total_bits = 0
            total_correct = 0
            with torch.no_grad():
                for _, inp_e, target_e in eval_batches:
                    inp_e = inp_e.to(DEVICE)  # (L+1,B,C)
                    target_e = target_e.to(DEVICE)  # (L,B,W)
                    L_e = target_e.shape[0]
                    B_e = target_e.shape[1]
                    C_e = inp_e.shape[2]
                    blanks_e = torch.zeros(L_e, B_e, C_e, device=DEVICE)
                    x_e = torch.cat([inp_e, blanks_e], dim=0)  # (2L+1,B,C)
                    out_full = model(x_e)  # (2L+1,B,W)
                    out_e = out_full[-L_e:]  # (L,B,W)
                    l = criterion(out_e, target_e)
                    total_loss += float(l.item())
                    pred_e = (out_e >= 0.5).float()
                    total_bits += int(target_e.numel())
                    total_correct += int((pred_e == target_e).sum().item())
            avg_loss = total_loss / max(1, len(eval_batches))
            bit_accuracy = total_correct / max(1, total_bits)
            wandb.log(
                {
                    "eval_loss": avg_loss,
                    "eval_bit_acc": bit_accuracy,
                    "step": step,
                }
            )

            # checkpoint best/last
            last_path = os.path.join(ckpt_dir, "gen2_last.pt")
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
                best_path = os.path.join(ckpt_dir, "gen2_best.pt")
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
