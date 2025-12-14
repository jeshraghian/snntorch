from datetime import datetime
import argparse
import re
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import wandb

from snntorch._neurons.leaky import Leaky


date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"output-{date_string}.txt"
with open(filename, "w") as f:
    f.write("\n")


# Hyperparameters
SEQ_LENGTH = 512
HIDDEN_DIM = 512
LR = 1e-3
EPOCHS = 10000
BATCH_SIZE = 64
CHUNKED_BATCH_SIZE = 16
LEARN_BETA = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECODE_EVERY_N_BATCHES = 50
print("Device: ", DEVICE)

# Model options

# Reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

def _decimal_float(value_str: str) -> float:
    if not re.fullmatch(r"[0-9]*\.[0-9]+", value_str):
        raise argparse.ArgumentTypeError(
            f"Invalid --lr '{value_str}'. Use decimal notation like 0.0005 (no scientific notation)."
        )
    val = float(value_str)
    if val <= 0.0:
        raise argparse.ArgumentTypeError("--lr must be > 0.")
    return val

def parse_args():
    parser = argparse.ArgumentParser(description="TinyStories Leaky training.")
    parser.add_argument(
        "--lr",
        type=_decimal_float,
        required=False,
        help="Learning rate in decimal notation (e.g., 0.0005). No scientific notation.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Optional wandb run name.",
    )
    return parser.parse_args()

ARGS = parse_args()


def initialize_wandb(run_name=None):
    wandb.init(
        project="snntorch-ssm",
        config={
            "seq_length": SEQ_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learn_beta": LEARN_BETA,
            "variant": "Leaky-stepwise",
        },
        name=run_name,
    )


# Load TinyStories dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split="train")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2"
)  # Use GPT-2 tokenizer for simplicity
tokenizer.pad_token = (
    tokenizer.eos_token
)  # Use the end-of-sequence token as padding

print("initialized tokenizer")

VOCAB_SIZE = tokenizer.vocab_size


def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
    )
    return {"input_ids": tokens["input_ids"]}


tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

print("tokenized dataset")

dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)


class SNNLanguageModelLeaky(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(SNNLanguageModelLeaky, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(SEQ_LENGTH, hidden_dim)
        self.lif1 = Leaky(
            beta=torch.full((hidden_dim,), 0.9, device=DEVICE),
            learn_beta=LEARN_BETA,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = Leaky(
            beta=torch.full((hidden_dim,), 0.9, device=DEVICE),
            learn_beta=LEARN_BETA,
        )
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lif3 = Leaky(
            beta=torch.full((hidden_dim,), 0.9, device=DEVICE),
            learn_beta=LEARN_BETA,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        # untied output head

    def forward(self, x):
        # x: [SEQ_LENGTH-1, B] token IDs (torch.long)
        T, B = x.shape
        mem1 = torch.zeros(B, HIDDEN_DIM, device=x.device)
        mem2 = torch.zeros(B, HIDDEN_DIM, device=x.device)
        mem3 = torch.zeros(B, HIDDEN_DIM, device=x.device)

        logits_list = []
        pos = torch.arange(T, device=x.device)
        pos_table = self.pos_embedding(pos)  # [T, HIDDEN_DIM]
        for t in range(T):
            hidden = self.embedding(x[t]) + pos_table[t]  # [B, hidden_dim]

            spk1, mem1 = self.lif1(hidden, mem1)

            hidden = self.fc2(spk1)
            hidden = torch.relu(hidden)
            spk2, mem2 = self.lif2(hidden, mem2)

            hidden = self.fc3(spk2)
            hidden = torch.relu(hidden)
            spk3, mem3 = self.lif3(hidden, mem3)
            output_t = self.fc_out(spk3)  # [B, vocab_size]
            logits_list.append(output_t)

        output = torch.stack(logits_list, dim=0)  # [T, B, vocab_size]
        return output


# Apply optional LR override before wandb init so config reflects it
if ARGS.lr is not None:
    LR = ARGS.lr
initialize_wandb(run_name=ARGS.name)

# Initialize model, loss, and optimizer
model = SNNLanguageModelLeaky(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(model.parameters(), lr=LR)


# Training Loop
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    batch_num = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        batch_num += 1
        x_ids = batch["input_ids"].to(DEVICE)

        # build mask up to and including the first EOS in the targets (no attention needed)
        eos_id = tokenizer.eos_token_id
        # targets are x_ids shifted left by 1: compare positions 1..S-1 to eos
        is_eos_shifted = x_ids[:, 1:] == eos_id  # [B, S-1]
        has_eos = is_eos_shifted.any(dim=1)  # [B]
        # if no EOS present, include all targets -> first_idx_y = (S-1)-1 = S-2
        first_idx_y = torch.full(
            (x_ids.size(0),), (SEQ_LENGTH - 2), device=DEVICE, dtype=torch.long
        )
        tmp_first = is_eos_shifted.float().argmax(dim=1).long()
        first_idx_y[has_eos] = tmp_first[has_eos]
        t = torch.arange(SEQ_LENGTH - 1, device=DEVICE).unsqueeze(
            1
        )  # [S-1, 1]
        y_mask = t <= first_idx_y.unsqueeze(0)  # [S-1, B]

        # process batch: token IDs / teacher forcing setup / permute to (seq_length, batch)
        x_tok = x_ids[:, :-1]  # Input token IDs: all but the last token
        y_tok = x_ids[:, 1:]  # Target token IDs: next token in the sequence
        x = x_tok.permute(1, 0)  # [SEQ_LENGTH-1, B]
        y_labels = y_tok.permute(1, 0)  # [SEQ_LENGTH-1, B]

        optimizer.zero_grad()

        # gradient accumulation over batch dimension using CHUNKED_BATCH_SIZE
        B_total = x.shape[1]
        total_valid_tokens = int(y_mask.sum().item())
        decode_this_batch = batch_num % DECODE_EVERY_N_BATCHES == 0
        have_already_decoded_this_batch = False
        total_loss_sum = 0.0
        mean_loss_samples = []

        for b_start in range(0, B_total, CHUNKED_BATCH_SIZE):
            b_end = min(b_start + CHUNKED_BATCH_SIZE, B_total)

            x_chunk = x[:, b_start:b_end]
            y_chunk_labels = y_labels[:, b_start:b_end]

            output_chunk = model(x_chunk)

            # occasional decoding from the first sample of the batch
            if (
                decode_this_batch
                and not have_already_decoded_this_batch
                and (b_end - b_start) > 0
            ):
                # Autoregressive greedy decode using the first sample in the chunk
                with torch.no_grad():
                    eos_id = tokenizer.eos_token_id
                    # seed with the first input token
                    seq_ids = x_chunk[0:1, 0:1].clone()  # [1,1]
                    gen_ids = []
                    for _ in range(SEQ_LENGTH - 1):
                        logits_all = model(seq_ids)
                        next_token = torch.argmax(logits_all[-1, 0, :], dim=-1)
                        gen_ids.append(int(next_token.item()))
                        if int(next_token.item()) == eos_id:
                            break
                        next_token_long = next_token.view(1, 1).to(
                            seq_ids.dtype
                        )
                        seq_ids = torch.cat([seq_ids, next_token_long], dim=0)
                with open(filename, "a") as f:
                    f.write(tokenizer.decode(gen_ids))
                    f.write("\n")
                have_already_decoded_this_batch = True

            # compute masked loss per chunk: include up to and including first EOS only
            flat_logits = output_chunk.reshape(-1, VOCAB_SIZE)
            flat_targets = y_chunk_labels.reshape(-1)
            flat_mask = y_mask[:, b_start:b_end].reshape(-1).bool()
            # Assume valid chunk and compute losses/stats directly
            loss_sum_chunk = F.cross_entropy(
                flat_logits[flat_mask],
                flat_targets[flat_mask],
                reduction="sum",
            )
            total_loss_sum += float(loss_sum_chunk.item())
            # Vectorized per-sample mean loss within this chunk (no grad, CPU)
            with torch.no_grad():
                per_token_loss = F.cross_entropy(
                    output_chunk.detach().reshape(-1, VOCAB_SIZE),
                    y_chunk_labels.reshape(-1),
                    reduction="none",
                ).reshape(output_chunk.shape[0], output_chunk.shape[1])  # [T,Bc]
                mask_chunk = y_mask[:, b_start:b_end].bool()  # [T,Bc]
                valid_counts = mask_chunk.sum(dim=0).clamp_min(1)  # [Bc]
                loss_sum_per_sample = (per_token_loss * mask_chunk).sum(dim=0)  # [Bc]
                mean_loss_per_sample = loss_sum_per_sample / valid_counts  # [Bc]
                mean_loss_samples.append(mean_loss_per_sample.cpu())
            if total_valid_tokens > 0:
                (loss_sum_chunk / float(total_valid_tokens)).backward()

        total_loss_mean = total_loss_sum / float(total_valid_tokens)
        # finite perplexity: clip loss to avoid overflow but never use inf
        ppl = math.exp(min(total_loss_mean, 20.0))
        # Per-batch error bars across sample-level perplexities (aggregated across chunks)
        mean_loss_all = torch.cat(mean_loss_samples, dim=0)  # [B] on CPU
        mean_loss_np = mean_loss_all.numpy()
        ppl_samples_np = np.exp(np.clip(mean_loss_np, None, 20.0))
        ppl_std = float(np.std(ppl_samples_np))
        ppl_min = float(np.min(ppl_samples_np))
        ppl_max = float(np.max(ppl_samples_np))
        global_step += 1
        wandb.log(
            {
                "loss": total_loss_mean,
                "ppl": ppl,
                "ppl_std": ppl_std,
                "ppl_min": ppl_min,
                "ppl_max": ppl_max,
                "step": global_step,
                "epoch": epoch + 1,
            }
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += total_loss_mean

    train_loss /= len(dataloader)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss: .4f}")

print("Training Complete!")
