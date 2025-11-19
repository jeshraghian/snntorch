from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import sys

from snntorch._neurons.gen2_unoptimized_mem_eff import Gen2SingleInputReadout


date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"output-{date_string}.txt"
with open(filename, "w") as f:
    f.write("\n")


# Hyperparameters
SEQ_LENGTH = 128
HIDDEN_DIM = 256
LR = 5e-3
EPOCHS = 10000
BATCH_SIZE = 64
CHUNKED_BATCH_SIZE = 8
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DECODE_EVERY_N_BATCHES = 50
INPUT_TOPK_TAU = 2.0
KEY_TOPK_TAU = 2.0
print("Device: ", DEVICE)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)


def initialize_wandb(run_name=None):
    wandb.init(
        project="snntorch-ssm",
        config={
            "seq_length": SEQ_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "variant": "Gen2",
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

data_loader = DataLoader(
    tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True
)


class SNNLanguageModelGen2(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(SNNLanguageModelGen2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(SEQ_LENGTH, hidden_dim)

        # Require perfect square hidden_dim to use the convenient constructor
        m = int(math.isqrt(hidden_dim))
        if m * m != hidden_dim:
            raise ValueError(
                f"HIDDEN_DIM must be a perfect square to use from_num_spiking_neurons; got {hidden_dim}"
            )

        # Choose Top-K values based on hidden_dim and n=m
        # input_topk = max(1, min(hidden_dim - 1, hidden_dim // 16))  # ~6.25%
        # input_topk = hidden_dim
        input_topk = None
        # key_topk = max(1, min(m - 1, m // 4))  # ~25% of n
        # key_topk = m
        key_topk = None

        # Gen2 layers produce (T, B, hidden_dim) with use_q_projection=False
        self.gen2_1 = Gen2SingleInputReadout.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            time_chunk_size=None,
            use_q_projection=False,
            input_topk=input_topk,
            key_topk=key_topk,
            input_topk_tau=INPUT_TOPK_TAU,
            key_topk_tau=KEY_TOPK_TAU,
        ).to(DEVICE)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gen2_2 = Gen2SingleInputReadout.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            time_chunk_size=None,
            use_q_projection=False,
            input_topk=input_topk,
            key_topk=key_topk,
            input_topk_tau=INPUT_TOPK_TAU,
            key_topk_tau=KEY_TOPK_TAU,
        ).to(DEVICE)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.gen2_3 = Gen2SingleInputReadout.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            time_chunk_size=None,
            use_q_projection=False,
            input_topk=input_topk,
            key_topk=key_topk,
            input_topk_tau=INPUT_TOPK_TAU,
            key_topk_tau=KEY_TOPK_TAU,
        ).to(DEVICE)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        T = x.size(0)
        hidden = self.embedding(x)  # [T, B, hidden_dim]
        pos = torch.arange(hidden.size(0), device=hidden.device)
        pos_table = self.pos_embedding(pos).unsqueeze(1)  # [T, 1, H]
        hidden = hidden + pos_table

        # Gen2 layer 1
        hidden, _ = self.gen2_1(hidden)  # (T, B, H)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # Nonlinear hidden
        hidden = self.fc2(hidden)
        hidden = torch.relu(hidden)
        hidden = hidden.reshape(T, -1, hidden.shape[-1])

        # Gen2 layer 2
        hidden, _ = self.gen2_2(hidden)  # (T, B, H)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # Nonlinear hidden
        hidden = self.fc3(hidden)
        hidden = torch.relu(hidden)
        hidden = hidden.reshape(T, -1, hidden.shape[-1])

        # Gen2 layer 3
        hidden, _ = self.gen2_3(hidden)  # (T, B, H)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # Output transformation
        output = self.fc_out(hidden)
        output = output.reshape(T, -1, output.shape[-1])
        return output


# Optional run name from first CLI arg
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else None
initialize_wandb(run_name=RUN_NAME)

# Initialize model, loss, and optimizer
model = SNNLanguageModelGen2(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    batch_num = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}"):
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
            num_valid_chunk = int(flat_mask.sum().item())
            if num_valid_chunk > 0:
                loss_sum_chunk = F.cross_entropy(
                    flat_logits[flat_mask],
                    flat_targets[flat_mask],
                    reduction="sum",
                )
                total_loss_sum += float(loss_sum_chunk.item())
                if total_valid_tokens > 0:
                    (loss_sum_chunk / float(total_valid_tokens)).backward()
            else:
                raise ValueError("No valid tokens in chunk")

        total_loss_mean = total_loss_sum / float(total_valid_tokens)
        ppl = (
            math.exp(total_loss_mean) if total_loss_mean < 20 else float("inf")
        )
        wandb.log({"loss": total_loss_mean, "ppl": ppl})
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += total_loss_mean

    train_loss /= len(data_loader)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss: .4f}")

print("Training Complete!")
