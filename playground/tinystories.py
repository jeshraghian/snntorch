import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from snntorch._neurons.stateleaky import StateLeaky
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from datetime import datetime

date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"output-{date_string}.txt"
with open(filename, "w") as f:
    f.write("\n")


# Hyperparameters
SEQ_LENGTH = 128
HIDDEN_DIM = 512
LR = 1e-3
EPOCHS = 10000
BATCH_SIZE = 64
CHUNKED_BATCH_SIZE = 8
LEARN_BETA = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DECODE_EVERY_N_BATCHES = 50
print("Device: ", DEVICE)


def initialize_wandb():
    wandb.init(
        project="snntorch-ssm",
        config={
            "seq_length": SEQ_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learn_beta": LEARN_BETA,
        },
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


class SNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(SNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lif1 = StateLeaky(
            beta=torch.tensor([0.9]).to(DEVICE),
            channels=hidden_dim,
            learn_beta=LEARN_BETA,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = StateLeaky(
            beta=torch.tensor([0.9]).to(DEVICE),
            channels=hidden_dim,
            learn_beta=LEARN_BETA,
        )
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lif3 = StateLeaky(
            beta=torch.tensor([0.9]).to(DEVICE),
            channels=hidden_dim,
            learn_beta=LEARN_BETA,
        )
        self.fc4 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [SEQ_LENGTH-1, B] token IDs (torch.long)
        hidden = self.embedding(x)  # [SEQ_LENGTH-1, B, hidden_dim]
        hidden, _ = self.lif1(hidden)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # nonlinear hidden
        hidden = self.fc2(hidden)
        hidden = torch.relu(hidden)
        hidden = hidden.reshape(SEQ_LENGTH - 1, -1, hidden.shape[-1])
        hidden, _ = self.lif2(hidden)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # nonlinear hidden
        hidden = self.fc3(hidden)
        hidden = torch.relu(hidden)
        hidden = hidden.reshape(SEQ_LENGTH - 1, -1, hidden.shape[-1])
        hidden, _ = self.lif3(hidden)
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # output transformation
        output = self.fc4(hidden)
        output = output.reshape(SEQ_LENGTH - 1, -1, output.shape[-1])
        return output


initialize_wandb()

# Initialize model, loss, and optimizer
model = SNNLanguageModel(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    batch_num = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        batch_num += 1
        # print("batch num: ", batch_num)
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
                # output_chunk: [SEQ_LENGTH-1, b_chunk, VOCAB_SIZE]
                seq_translate = torch.argmax(output_chunk[:, 0, :], dim=-1)
                assert seq_translate.shape[0] == SEQ_LENGTH - 1
                with open(filename, "a") as f:
                    f.write(tokenizer.decode(seq_translate.tolist()))
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
        wandb.log({"loss": total_loss_mean})
        optimizer.step()

        train_loss += total_loss_mean

    train_loss /= len(dataloader)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss: .4f}")

print("Training Complete!")
