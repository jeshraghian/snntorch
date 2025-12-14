import torch
import torch.nn as nn
from snntorch._neurons.associative import AssociativeLeaky

SEQ_LENGTH = 512
HIDDEN_DIM = 256


class SpikingAssocBlock(nn.Module):
    """AssociativeLeaky + Linear + ReLU, applied over (T, B, H)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Delegate validation to AssociativeLeaky; it enforces perfect-square when using this ctor
        self.assoc = AssociativeLeaky.from_num_spiking_neurons(
            in_dim=hidden_dim,
            num_spiking_neurons=hidden_dim,
            use_q_projection=True,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, H)
        y = self.assoc(x)  # (T, B, H)
        h = self.fc(y)  # (T, B, H)
        h = self.act(h)
        return h


class SNNAssociativeLanguageModel(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dim: int = HIDDEN_DIM, n_layers: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(SEQ_LENGTH, hidden_dim)
        self.blocks = nn.ModuleList(
            [SpikingAssocBlock(hidden_dim) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (T, B)
        T, B = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.token_emb(tokens)  # (T, B, H)
        h = h + self.pos_emb(pos).unsqueeze(1)  # (T, 1, H) broadcast over B
        for block in self.blocks:
            h = block(h)  # (T, B, H)
        logits = self.out_proj(h)  # (T, B, vocab_size)
        return logits


if __name__ == "__main__":
    vocab_size = 120
    T, B = 8, 2
    tokens = torch.randint(0, vocab_size, (T, B))
    model = SNNAssociativeLanguageModel(
        vocab_size=vocab_size, hidden_dim=64, n_layers=2
    )
    logits = model(tokens)
    print("assoc_llm demo:", logits.shape)
