# The small Transformer of the LLM part of the course (L1.x onward). The model grows from notebook to notebook:
# each notebook adds an ingredient to the code of this module. It works on the data of toyworld.py.
import numpy as np


def sinusoid(T, d):
    # sinusoidal positions of Vaswani et al. (2017), as in L0.9: one row per position, d components
    t = np.arange(T)[:, None]
    omega = 1 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(t * omega)
    pe[:, 1::2] = np.cos(t * omega)
    return pe


# ---- L1.0: attention only. Embedding + position, then layers of attention with a residual connection.
# Later notebooks add the other ingredients (heads, MLP, layer normalization) to the same classes.
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    # softmax(Q K^T / sqrt(d_k)) V with a causal mask; W_Q, W_K, W_V in one matrix, W_O to mix the output
    def __init__(self, d, heads=1):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.identity = False              # if True each position reads only itself (used to remove the attention)
        self.weights = None                # attention weights, kept when keep_weights=True: (batch, heads, T, T)

    def forward(self, x, keep_weights=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if keep_weights or self.identity:
            # the computation written out, to read the weights (the same as the fused function of the else branch)
            scores = q @ k.transpose(-1, -2) / (D // self.heads) ** 0.5
            scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
            weights = scores.softmax(dim=-1)
            if self.identity:
                weights = torch.eye(T).expand_as(weights)
            self.weights = weights.detach()
            out = weights @ v
        else:
            out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)      # faster in training
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class AttentionLM(nn.Module):
    # x_t = embedding(word_t) + position_t; each layer: x <- x + attention(x); scores = x W_U.
    # With previous_token=True the embedding of the previous word is added to the input too (an experiment of L1.0).
    def __init__(self, vocabulary_size, d=32, layers=2, heads=1, max_length=32, previous_token=False):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d)
        self.position = nn.Embedding(max_length, d)
        self.previous = nn.Embedding(vocabulary_size, d) if previous_token else None
        self.layers = nn.ModuleList([SelfAttention(d, heads) for _ in range(layers)])
        self.unembed = nn.Linear(d, vocabulary_size)

    def forward(self, tokens, keep_weights=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
        if self.previous is not None:
            x = x + self.previous(nn.functional.pad(tokens, (1, 0))[:, :-1])
        for layer in self.layers:
            x = x + layer(x, keep_weights)
        return self.unembed(x)

    def attention_maps(self, tokens):
        # the attention weights of each layer, (batch, heads, T, T), for a batch of sequences
        with torch.no_grad():
            self(tokens, keep_weights=True)
        return [layer.weights for layer in self.layers]

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def answer_scores(model, sequences):
    # recall sequences end with `person drinks answer </s>`: the scores of the answer come from the position of `drinks`
    return model(sequences[:, :-2])[:, -1]


def answer_accuracy(model, sequences):
    with torch.no_grad():
        return (answer_scores(model, sequences).argmax(dim=-1) == sequences[:, -2]).float().mean().item()


def fit_recall(sequences, seed=0, steps=20000, lr=3e-3, batch_size=128, **model_kwargs):
    # trains an AttentionLM on the answer word of the recall sequences; returns the model, with the loss averaged over each
    # block of 100 steps in model.loss_curve. `sequences` is a tensor of word indices or a list of such tensors, one for
    # each length of the list (each batch is drawn from one of them, so that it is rectangular)
    groups = [sequences] if torch.is_tensor(sequences) else list(sequences)
    torch.manual_seed(seed)
    model = AttentionLM(max(int(g.max()) for g in groups) + 1, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        group = groups[0] if len(groups) == 1 else groups[torch.randint(len(groups), (1,), generator=generator).item()]
        batch = group[torch.randint(len(group), (batch_size,), generator=generator)]
        loss = nn.functional.cross_entropy(answer_scores(model, batch), batch[:, -2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1)
    return model
