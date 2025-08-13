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
LEARN_BETA = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)


def initialize_wandb():
    wandb.init(project="snntorch-ssm", config={
        "seq_length": SEQ_LENGTH,
        "hidden_dim": HIDDEN_DIM,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learn_beta": LEARN_BETA,
    })


# Load TinyStories dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split="train")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer for simplicity
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding

print("initialized tokenizer")

VOCAB_SIZE = tokenizer.vocab_size


def tokenize_fn(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=SEQ_LENGTH, padding="max_length")
    return {"input_ids": tokens["input_ids"]}


tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

print("tokenized dataset")

dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)


class SNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(SNNLanguageModel, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.lif1 = StateLeaky(beta=torch.tensor([0.9]).to(
            DEVICE), learn_decay_filter=False, channels=hidden_dim, learn_beta=LEARN_BETA)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = StateLeaky(beta=torch.tensor([0.9]).to(
            DEVICE), learn_decay_filter=False, channels=hidden_dim, learn_beta=LEARN_BETA)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lif3 = StateLeaky(beta=torch.tensor([0.9]).to(
            DEVICE), learn_decay_filter=False, channels=hidden_dim, learn_beta=LEARN_BETA)
        self.fc4 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])

        # input transformation
        hidden = self.fc1(x)
        hidden = hidden.reshape(SEQ_LENGTH - 1, -1, hidden.shape[-1])
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
        x = batch["input_ids"].to(DEVICE)

        # process batch: one hot / teacher forcing setup / permute to (seq_length, batch, vocab_size)
        x = F.one_hot(x, num_classes=VOCAB_SIZE).float()
        y = x[:, 1:]  # Target: next token in the sequence
        x = x[:, :-1]  # Input: all but the last token
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        optimizer.zero_grad()

        output = model(x)

        # print the decoding
        if batch_num % 50 == 0:
            output = output.permute(1, 0, 2)
            seq_translate = torch.argmax(output[0], dim=-1)
            assert seq_translate.shape[0] == SEQ_LENGTH - 1
            with open(filename, "a") as f:
                f.write(tokenizer.decode(seq_translate))
                f.write("\n")
            output = output.permute(1, 0, 2)

        # assert output.shape == (SEQ_LENGTH-1, BATCH_SIZE, VOCAB_SIZE)
        # assert y.shape == (SEQ_LENGTH-1, BATCH_SIZE, VOCAB_SIZE)

        y = y.argmax(dim=-1)
        loss = criterion(output.reshape(-1, VOCAB_SIZE), y.reshape(-1))  # Compute loss
        wandb.log({"loss": loss.item()})

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss: .4f}")

print("Training Complete!")
