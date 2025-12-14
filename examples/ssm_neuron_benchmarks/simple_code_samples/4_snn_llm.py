import torch
import torch.nn as nn
from snntorch._neurons.stateleaky import StateLeaky

SEQ_LENGTH = 512
HIDDEN_DIM = 256
LEARN_BETA = True


class SpikingBlock(nn.Module):
    """StateLeaky + Linear + ReLU, applied over (T, B, H)."""

    def __init__(self, hidden_dim):
        super().__init__()
        beta0 = torch.full((hidden_dim,), 0.9)
        self.ssm = StateLeaky(
            beta=beta0,
            channels=hidden_dim,
            learn_beta=LEARN_BETA,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (T, B, H)
        spk, v = self.ssm(x)  # (T, B, H)
        h = self.fc(v)  # (T, B, H)
        h = self.act(h)
        return h


class SNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=HIDDEN_DIM, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(SEQ_LENGTH, hidden_dim)

        self.blocks = nn.ModuleList(
            [SpikingBlock(hidden_dim) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        T, B = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.token_emb(tokens)  # (T, B, H)
        h = h + self.pos_emb(pos).unsqueeze(1)  # (T, 1, H) broadcast over B

        for block in self.blocks:
            h = block(h)  # (T, B, H)

        logits = self.out_proj(h)  # (T, B, vocab_size)
        return logits


if __name__ == "__main__":
    vocab_size = 100
    T, B = 8, 2
    tokens = torch.randint(0, vocab_size, (T, B))
    model = SNNLanguageModel(vocab_size=vocab_size, hidden_dim=64, n_layers=2)
    logits = model(tokens)
    print("snn_llm demo:", logits.shape)
