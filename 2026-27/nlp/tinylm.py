# The small Transformer of the LLM part of the course (L1.x onward). The model grows from notebook to notebook:
# each notebook adds an ingredient to the code of this module. It works on the data of toyworld.py.
import copy
import os

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
        self.silenced = []                 # L1.1: heads whose output is set to zero (used to remove one head)
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
        if self.silenced:
            out = out.clone()
            out[:, self.silenced] = 0
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

    def stream(self, tokens):
        # L1.1: the final vector of each position is a SUM of what each component wrote in it. Returns the terms, in order:
        # (name, tensor (batch, T, d)); their sum is the vector that the output layer reads
        with torch.no_grad():
            x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
            if self.previous is not None:
                x = x + self.previous(nn.functional.pad(tokens, (1, 0))[:, :-1])
            terms = [("embedding + position", x)]
            for k, layer in enumerate(self.layers):
                terms.append((f"attention of layer {k + 1}", layer(x)))
                x = x + terms[-1][1]
        return terms

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def answer_scores(model, sequences):
    # recall sequences end with `person drinks answer </s>`: the scores of the answer come from the position of `drinks`
    return model(sequences[:, :-2])[:, -1]


def answer_accuracy(model, sequences):
    with torch.no_grad():
        return (answer_scores(model, sequences).argmax(dim=-1) == sequences[:, -2]).float().mean().item()


def fit_recall(sequences, seed=0, steps=20000, lr=3e-3, batch_size=128, model_class=None, **model_kwargs):
    # trains an AttentionLM on the answer word of the recall sequences; returns the model, with the loss averaged over each
    # block of 100 steps in model.loss_curve. `sequences` is a tensor of word indices or a list of such tensors, one for
    # each length of the list (each batch is drawn from one of them, so that it is rectangular)
    groups = [sequences] if torch.is_tensor(sequences) else list(sequences)
    torch.manual_seed(seed)
    model = (model_class or AttentionLM)(max(int(g.max()) for g in groups) + 1, **model_kwargs)
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


# ---- L1.1: the Transformer block. Multi-head attention (SelfAttention above, heads=H), an MLP and layer normalization,
# arranged around the residual stream.
class MLP(nn.Module):
    # applied to each position on its own: d -> mult*d -> d, with a non-linearity
    def __init__(self, d, mult=4):
        super().__init__()
        self.up = nn.Linear(d, mult * d)
        self.down = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.down(nn.functional.gelu(self.up(x)))


class Block(nn.Module):
    # norm="pre":  x <- x + attention(LN(x));  x <- x + MLP(LN(x))     (GPT-2 and most recent models)
    # norm="post": x <- LN(x + attention(x));  x <- LN(x + MLP(x))     (the original Transformer)
    # norm="none": no normalization. mlp=0 removes the MLP (the layer of L1.0).
    def __init__(self, d, heads=1, mlp=4, norm="pre"):
        super().__init__()
        self.norm = norm
        self.attention = SelfAttention(d, heads)
        self.mlp = MLP(d, mlp) if mlp else None
        self.ln1 = nn.LayerNorm(d) if norm != "none" else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if norm != "none" and mlp else nn.Identity()

    def forward(self, x, keep_weights=False, terms=None):
        post = self.norm == "post"
        a = self.attention(x if post else self.ln1(x), keep_weights)
        if terms is not None:
            terms.append(("attention", a))
        x = self.ln1(x + a) if post else x + a
        if self.mlp is not None:
            m = self.mlp(x if post else self.ln2(x))
            if terms is not None:
                terms.append(("MLP", m))
            x = self.ln2(x + m) if post else x + m
        return x


class TinyTransformer(nn.Module):
    # embedding + position, a stack of blocks, (final LayerNorm for pre-norm), output layer. AttentionLM is the special
    # case heads=1, mlp=0, norm="none".
    def __init__(self, vocabulary_size, d=32, layers=2, heads=1, mlp=4, norm="pre", max_length=32):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d)
        self.position = nn.Embedding(max_length, d)
        self.layers = nn.ModuleList([Block(d, heads, mlp, norm) for _ in range(layers)])
        self.final = nn.LayerNorm(d) if norm == "pre" else nn.Identity()
        self.unembed = nn.Linear(d, vocabulary_size)

    def forward(self, tokens, keep_weights=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
        for block in self.layers:
            x = block(x, keep_weights)
        return self.unembed(self.final(x))

    def attention_maps(self, tokens):
        with torch.no_grad():
            self(tokens, keep_weights=True)
        return [block.attention.weights for block in self.layers]

    def stream(self, tokens):
        # the terms written in the residual stream (as AttentionLM.stream, plus the MLPs): exact for norm="none", where the
        # output layer is applied to the sum of the terms
        terms = []
        with torch.no_grad():
            x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
            out = [("embedding + position", x)]
            for k, block in enumerate(self.layers):
                block_terms = []
                x = block(x, terms=block_terms)
                out += [(f"{name} {k + 1}", t) for name, t in block_terms]
        return out

    def norms(self, tokens):
        # the length of the vectors of the residual stream, averaged over positions, at the entrance and after each block
        with torch.no_grad():
            x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
            out = [x.norm(dim=-1).mean().item()]
            for block in self.layers:
                x = block(x)
                out.append(x.norm(dim=-1).mean().item())
        return out

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---- L1.1: a language model on all the tokens, and autoregressive generation
def fit_lm(sequences, seed=0, steps=6000, lr=3e-3, batch_size=128, model_class=None, callback=None, checkpoints=(), vocabulary_size=None, **model_kwargs):
    # next-token prediction on every position (the padding is ignored); `sequences` is a tensor of word indices
    torch.manual_seed(seed)
    model = (model_class or TinyTransformer)(vocabulary_size or int(sequences.max()) + 1, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        batch = sequences[torch.randint(len(sequences), (batch_size,), generator=generator)]
        logits = model(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
        if callback is not None and len(losses) in checkpoints:
            callback(len(losses), model)                    # L1.4: measure something while the network trains
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1)
    return model


def next_word_probabilities(model, prefix):
    # the distribution of the next word given a list of word indices
    with torch.no_grad():
        return model(torch.tensor([prefix]))[0, -1].softmax(dim=-1)


def generate(model, prefix, steps, temperature=1.0, seed=0, stop=None):
    # autoregressive generation: at each step the model gives a distribution over the next word, one word is chosen
    # (temperature 0 = the most probable, greater = sampling) and appended to the text. Returns the final list of word indices
    # and the distribution used at each step (steps x vocabulary).
    rng = np.random.default_rng(seed)
    text, distributions = list(prefix), []
    for _ in range(steps):
        probabilities = next_word_probabilities(model, text)
        distributions.append(probabilities)
        if temperature == 0:
            word = int(probabilities.argmax())
        else:
            p = (probabilities.log() / temperature).softmax(dim=-1).numpy().astype("float64")
            word = int(rng.choice(len(p), p=p / p.sum()))
        text.append(word)
        if stop is not None and word == stop:
            break
    return text, torch.stack(distributions)


# ---- L1.2: encoder and decoder. The attention takes an explicit mask (a matrix of allowed pairs) and, for the cross-attention,
# keys and values that come from another sequence.
class MultiHeadAttention(nn.Module):
    # softmax(Q K^T / sqrt(d_k)) V where Q comes from `x` and K, V from `memory` (memory = x for self-attention).
    # `allowed` is a boolean matrix, True where the position of x may read the position of memory: (T_x, T_mem) or (B, T_x, T_mem).
    def __init__(self, d, heads=1):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.weights = None                # (batch, heads, T_x, T_mem), kept when keep_weights=True

    def forward(self, x, memory, allowed, keep_weights=False):
        B, Tx, D = x.shape
        Tm = memory.shape[1]
        q = self.q(x).view(B, Tx, self.heads, D // self.heads).transpose(1, 2)
        k, v = self.kv(memory).view(B, Tm, 2, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        allowed = allowed[:, None] if allowed.dim() == 3 else allowed[None, None]
        if keep_weights:
            scores = (q @ k.transpose(-1, -2) / (D // self.heads) ** 0.5).masked_fill(~allowed, float("-inf"))
            self.weights = scores.softmax(dim=-1).detach()
            out = self.weights @ v
        else:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.out(out.transpose(1, 2).reshape(B, Tx, D))


class EncoderBlock(nn.Module):
    # pre-norm block whose attention reads the whole source (no causal mask)
    def __init__(self, d, heads=1, mlp=4):
        super().__init__()
        self.attention = MultiHeadAttention(d, heads)
        self.mlp = MLP(d, mlp)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, allowed, keep_weights=False):
        h = self.ln1(x)
        x = x + self.attention(h, h, allowed, keep_weights)
        return x + self.mlp(self.ln2(x))


class DecoderBlock(nn.Module):
    # pre-norm block with three sub-layers: causal self-attention, cross-attention to the encoder, MLP.
    # cross=False removes the cross-attention (the decoder does not see the source).
    def __init__(self, d, heads=1, mlp=4, cross=True):
        super().__init__()
        self.self_attention = MultiHeadAttention(d, heads)
        self.cross_attention = MultiHeadAttention(d, heads) if cross else None
        self.mlp = MLP(d, mlp)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, memory, causal, source_allowed, keep_weights=False):
        h = self.ln1(x)
        x = x + self.self_attention(h, h, causal, keep_weights)
        if self.cross_attention is not None:
            x = x + self.cross_attention(self.ln2(x), memory, source_allowed, keep_weights)
        return x + self.mlp(self.ln3(x))


class Seq2Seq(nn.Module):
    # encoder: source embedding + position, blocks without mask; decoder: target embedding + position, blocks with causal
    # self-attention and cross-attention to the final vectors of the encoder; output layer on the target vocabulary.
    # forward(source, target_in) gives the scores of the next target word at each position of target_in.
    # cross=False removes the cross-attention (the decoder does not see the source); decoder_layers sets a different depth.
    def __init__(self, source_size, target_size, d=32, layers=2, heads=2, mlp=4, max_length=32, cross=True, decoder_layers=None):
        super().__init__()
        self.source_embedding, self.source_position = nn.Embedding(source_size, d), nn.Embedding(max_length, d)
        self.target_embedding, self.target_position = nn.Embedding(target_size, d), nn.Embedding(max_length, d)
        self.encoder = nn.ModuleList([EncoderBlock(d, heads, mlp) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d, heads, mlp, cross) for _ in range(decoder_layers or layers)])
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.unembed = nn.Linear(d, target_size)

    def encode(self, source, keep_weights=False):
        T = source.shape[1]
        real = (source != 0)                                                           # <pad> has index 0
        allowed = real[:, None, :].expand(-1, T, -1)                                    # nobody reads a <pad>
        allowed = allowed | torch.eye(T, dtype=torch.bool)                              # a position always reads itself (no empty rows)
        x = self.source_embedding(source) + self.source_position(torch.arange(T))
        for block in self.encoder:
            x = block(x, allowed, keep_weights)
        return self.encoder_norm(x), real

    def decode(self, memory, real, target_in, keep_weights=False):
        T = target_in.shape[1]
        causal = ~torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        source_allowed = real[:, None, :].expand(-1, T, -1)
        x = self.target_embedding(target_in) + self.target_position(torch.arange(T))
        for block in self.decoder:
            x = block(x, memory, causal, source_allowed, keep_weights)
        return self.unembed(self.decoder_norm(x))

    def forward(self, source, target_in, keep_weights=False):
        memory, real = self.encode(source, keep_weights)
        return self.decode(memory, real, target_in, keep_weights)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def fit_translation(source, target, seed=0, steps=4000, lr=3e-3, batch_size=64, **model_kwargs):
    # teacher forcing: the decoder reads the target shifted by one (<s> w1 ... wn) and predicts (w1 ... wn </s>), all positions
    torch.manual_seed(seed)
    model = Seq2Seq(int(source.max()) + 1, int(target.max()) + 1, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        i = torch.randint(len(source), (batch_size,), generator=generator)
        logits = model(source[i], target[i][:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target[i][:, 1:].reshape(-1), ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1) if steps % 100 == 0 else np.array(losses)
    return model


def translate(model, source, max_length=32, bos=1, eos=2):
    # greedy autoregressive decoding of a batch of sources: the encoder runs once, the decoder one word at a time
    with torch.no_grad():
        memory, real = model.encode(source)
        out = torch.full((len(source), 1), bos)
        for _ in range(max_length - 1):
            word = model.decode(memory, real, out)[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, word], dim=1)
            if ((out == eos).any(dim=1)).all():
                break
    return out


def translation_accuracy(model, source, target):
    # the share of sentences translated exactly, and of target words right at the right place (after the <s>)
    out = translate(model, source, max_length=target.shape[1])
    n = min(out.shape[1], target.shape[1])
    exact, words = [], []
    for o, t in zip(out[:, :n], target[:, :n]):
        stop = (t == 2).nonzero()[0].item() + 1
        exact.append(bool((o[:stop] == t[:stop]).all()))
        words.append((o[1:stop] == t[1:stop]).float().mean().item())
    return float(np.mean(exact)), float(np.mean(words))


# ---- L1.3: BERT. An encoder whose attention reads both sides, trained to fill in masked words (MLM) and then to classify.
MASK_ID, FIRST_WORD = 4, 6            # <mask> and the first index that is not a special token


class TinyBERT(nn.Module):
    # embeddings + position, EncoderBlock x layers (the blocks of L1.2, nobody reads a <pad>), a final LayerNorm and two heads:
    # `mlm_head` gives the scores of every word at every position, `class_head` the scores of the classes from the vector of the
    # first position (`<s>` plays the role of [CLS]). causal=True adds the triangular mask: only to compare with a decoder.
    def __init__(self, vocabulary_size, d=32, layers=2, heads=2, mlp=4, max_length=32, classes=2, causal=False):
        super().__init__()
        self.causal = causal
        self.embedding, self.position = nn.Embedding(vocabulary_size, d), nn.Embedding(max_length, d)
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, mlp) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.mlm_head, self.class_head = nn.Linear(d, vocabulary_size), nn.Linear(d, classes)

    def allowed(self, tokens):
        T = tokens.shape[1]
        allowed = (tokens != 0)[:, None, :].expand(-1, T, -1) | torch.eye(T, dtype=torch.bool)    # nobody reads a <pad>
        return allowed & ~torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if self.causal else allowed

    def encode(self, tokens, keep_weights=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
        allowed = self.allowed(tokens)
        for block in self.blocks:
            x = block(x, allowed, keep_weights)
        return self.norm(x)

    def forward(self, tokens, keep_weights=False):
        return self.mlm_head(self.encode(tokens, keep_weights))

    def classify(self, tokens):
        return self.class_head(self.encode(tokens)[:, 0])

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def mask_tokens(tokens, generator, rate=0.15, vocabulary_size=None, only_mask=False):
    # BERT's corruption: each word (not a special token) is picked with probability `rate`; a picked word becomes <mask> 80% of
    # the time, a random word 10%, and stays as it is 10%. Returns the corrupted input and the labels (-100 where no loss).
    pick = (torch.rand(tokens.shape, generator=generator) < rate) & (tokens >= FIRST_WORD)
    roll = torch.rand(tokens.shape, generator=generator)
    inputs = tokens.clone()
    inputs[pick & (roll < (2 if only_mask else 0.8))] = MASK_ID
    swap = pick & (roll >= 0.8) & (roll < 0.9) & (not only_mask)
    inputs[swap] = torch.randint(FIRST_WORD, vocabulary_size or int(tokens.max()) + 1, (int(swap.sum()),), generator=generator)
    return inputs, torch.where(pick, tokens, torch.full_like(tokens, -100))


def fit_mlm(tokens, vocabulary_size, seed=0, steps=3000, lr=3e-3, batch_size=128, callback=None, checkpoints=(), **model_kwargs):
    # pre-training: at every step a batch of documents is corrupted again (new masks each time), loss on the picked words only
    torch.manual_seed(seed)
    model = TinyBERT(vocabulary_size, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        i = torch.randint(len(tokens), (batch_size,), generator=generator)
        inputs, labels = mask_tokens(tokens[i], generator, vocabulary_size=vocabulary_size)
        loss = nn.functional.cross_entropy(model(inputs).reshape(-1, vocabulary_size), labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
        if callback is not None and len(losses) in checkpoints:
            callback(len(losses), model)
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1) if steps % 100 == 0 else np.array(losses)
    return model


def fit_classifier(tokens, labels, vocabulary_size, init=None, freeze=False, seed=0, steps=1500, lr=3e-3, batch_size=64, **model_kwargs):
    # fine-tuning: the same encoder, a loss on the class of the whole document read from the first position.
    # init = state of a pre-trained TinyBERT (None: from scratch); freeze=True trains the head only.
    torch.manual_seed(seed)
    model = TinyBERT(vocabulary_size, **model_kwargs)
    if init is not None:
        model.load_state_dict(init)
    optimizer = torch.optim.AdamW(model.class_head.parameters() if freeze else model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(tokens), (batch_size,), generator=generator)
        loss = nn.functional.cross_entropy(model.classify(tokens[i]), labels[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return model


def class_accuracy(model, tokens, labels):
    with torch.no_grad():
        return (model.classify(tokens).argmax(dim=-1) == labels).float().mean().item()


def fill_blank(model, tokens, positions):
    # hide one word per document (position[i] of document i) and return the model's best word and its distribution
    inputs = tokens.clone()
    rows = torch.arange(len(tokens))
    inputs[rows, positions] = MASK_ID
    with torch.no_grad():
        scores = model(inputs)[rows, positions].softmax(dim=-1)
    return scores.argmax(dim=-1), scores


# ---- L1.4: GPT. One stack of blocks that reads a single sequence `prefix <sep> continuation`, trained to write the continuation.
class DecoderOnly(nn.Module):
    # embeddings + position, EncoderBlock x layers whose mask is chosen here, a final LayerNorm and an output layer. With
    # reading="causal" (GPT) every position reads the ones on its left, the prefix included; "bidirectional" lets the positions of
    # the prefix (up to and including `sep`) read each other on both sides, the continuation stays causal (prefix-LM); "isolated"
    # lets the positions of the prefix read only themselves, the continuation reads everything on its left.
    def __init__(self, vocabulary_size, d=32, layers=4, heads=2, mlp=4, max_length=64, reading="causal", sep=5):
        super().__init__()
        self.reading, self.sep = reading, sep
        self.embedding, self.position = nn.Embedding(vocabulary_size, d), nn.Embedding(max_length, d)
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, mlp) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, vocabulary_size)

    def allowed(self, tokens):
        # (batch, T, T): True where the row may read the column
        B, T = tokens.shape
        causal = (~torch.triu(torch.ones(T, T, dtype=torch.bool), 1))[None].expand(B, -1, -1)
        if self.reading == "causal":
            return causal
        inside = torch.arange(T)[None] <= (tokens == self.sep).int().argmax(dim=1)[:, None]      # <s> source <sep>
        both = inside[:, :, None] & inside[:, None, :]
        if self.reading == "bidirectional":
            return causal | both
        return (causal & ~both) | (both & torch.eye(T, dtype=torch.bool)[None])

    def forward(self, tokens, keep_weights=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1]))
        allowed = self.allowed(tokens)
        for block in self.blocks:
            x = block(x, allowed, keep_weights)
        return self.unembed(self.norm(x))

    def attention_maps(self, tokens):
        with torch.no_grad():
            self(tokens, keep_weights=True)
        return [block.attention.weights for block in self.blocks]

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def fit_continuation(sequences, sep, seed=0, steps=1500, lr=3e-3, batch_size=64, model_class=None, vocabulary_size=None, **model_kwargs):
    # next-word prediction where the loss counts only the words after `sep` (the source is read, not predicted)
    torch.manual_seed(seed)
    model = (model_class or DecoderOnly)(vocabulary_size or int(sequences.max()) + 1, sep=sep, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    start = (sequences == sep).int().argmax(dim=1)
    losses = []
    for _ in range(steps):
        i = torch.randint(len(sequences), (batch_size,), generator=generator)
        x = sequences[i]
        target = x[:, 1:].clone()
        target[torch.arange(target.shape[1])[None] < start[i][:, None]] = 0          # position t predicts x[t + 1]: from `sep` on
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1) if steps % 100 == 0 else np.array(losses)
    return model


def complete(model, sequences, sep, eos=2):
    # greedy writing of the continuation of each `prefix <sep>`. The sequences are grouped by the length of the prefix (the
    # positions are absolute, a padded batch would shift them). Returns a list of tensors, one per sequence.
    start = (sequences == sep).int().argmax(dim=1)
    out = [None] * len(sequences)
    longest = model.position.num_embeddings
    with torch.no_grad():
        for length in start.unique().tolist():
            rows = (start == length).nonzero().flatten()
            x = sequences[rows][:, : length + 1]
            while x.shape[1] < min(sequences.shape[1], longest):
                x = torch.cat([x, model(x)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
                if (x == eos).any(dim=1).all():
                    break
            for j, r in enumerate(rows.tolist()):
                out[r] = x[j]
    return out


def completion_accuracy(model, sequences, sep, eos=2):
    # the share of continuations that are exactly right and of words right at the right place
    written = complete(model, sequences, sep, eos)
    exact, words = [], []
    for w, t in zip(written, sequences):
        first, last = (t == sep).nonzero()[0].item() + 1, (t == eos).nonzero()[0].item() + 1
        right, mine = t[first:last], w[first:first + (last - first)]
        n = min(len(right), len(mine))
        exact.append(len(mine) == len(right) and bool((mine == right).all()))
        words.append((mine[:n] == right[:n]).sum().item() / len(right))
    return float(np.mean(exact)), float(np.mean(words))


def hidden_states(model, tokens):
    # the vectors at the entrance of the network (word + position) and after each block, for every position:
    # a list of (batch, T, d) tensors; the last one is the output of the final normalization, the one that the output layer reads
    with torch.no_grad():
        if isinstance(model, TinyBERT):
            x = model.embedding(tokens) + model.position(torch.arange(tokens.shape[1]))
            allowed, blocks, final = model.allowed(tokens), model.blocks, model.norm
            run = lambda block, x: block(x, allowed)
        elif isinstance(model, DecoderOnly):
            x = model.embedding(tokens) + model.position(torch.arange(tokens.shape[1]))
            allowed, blocks, final = model.allowed(tokens), model.blocks, model.norm
            run = lambda block, x: block(x, allowed)
        else:                                                                          # TinyTransformer
            x = model.embedding(tokens) + model.position(torch.arange(tokens.shape[1]))
            blocks, final = model.layers, model.final
            run = lambda block, x: block(x)
        states = [x]
        for block in blocks:
            x = run(block, x)
            states.append(x)
        states[-1] = final(x)
    return states


def fit_targets(sequences, targets, seed=0, steps=3000, lr=3e-3, batch_size=128, model_class=None, init=None, vocabulary_size=None, **model_kwargs):
    # next-word prediction where the loss counts only the positions marked in `targets` (a 0/1 tensor with the shape of `sequences`:
    # targets[i, t] = 1 when the word t of sequence i is a word to predict). `init` is a state dict to start from (a pre-trained network).
    torch.manual_seed(seed)
    model = (model_class or TinyTransformer)(vocabulary_size or int(sequences.max()) + 1, **model_kwargs)
    if init is not None:
        model.load_state_dict(init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        i = torch.randint(len(sequences), (batch_size,), generator=generator)
        x, t = sequences[i], targets[i][:, 1:].bool()
        labels = x[:, 1:].masked_fill(~t, -100)
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1) if steps % 100 == 0 else np.array(losses)
    return model


# ---- L1.4: choosing the next word, and the key-value cache
def choose_next(scores, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    # one word per row of `scores` (batch, vocabulary): temperature, then top-k (keep the k best, 0 = all), then top-p (keep the
    # smallest set of best words whose probability reaches p), then sampling. temperature=0 gives the most probable word.
    if temperature == 0:
        return scores.argmax(dim=-1)
    scores = scores / temperature
    if top_k:
        kth = scores.topk(top_k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        order = scores.argsort(dim=-1, descending=True)
        sorted_probs = scores.gather(-1, order).softmax(dim=-1)
        before = sorted_probs.cumsum(dim=-1) - sorted_probs                  # probability of the words that come first
        drop = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, order, before >= top_p)
        scores = scores.masked_fill(drop, float("-inf"))
    return torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)[:, 0]


def write_texts(model, prefix, n, steps, seed=0, eos=2, **options):
    # n texts that continue the same prefix (a list of word indices), written together; after `eos` a text is filled with <pad> (0)
    generator = torch.Generator().manual_seed(seed)
    steps = min(steps, model.position.num_embeddings - len(prefix))             # the positions of the network end here
    texts = torch.tensor([prefix] * n)
    finished = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(steps):
            word = choose_next(model(texts)[:, -1], generator=generator, **options)
            word = word.masked_fill(finished, 0)
            texts = torch.cat([texts, word[:, None]], dim=1)
            finished |= word == eos
            if finished.all():
                break
    return texts


def beam_search(model, prefix, width=4, steps=40, eos=2):
    # the text with the highest total log-probability found by keeping the `width` best partial texts at each step.
    # Returns (list of word indices, log-probability).
    beams = [(list(prefix), 0.0)]
    done = []
    steps = min(steps, model.position.num_embeddings - len(prefix))
    with torch.no_grad():
        for _ in range(steps):
            candidates = []
            for text, score in beams:
                log_p = model(torch.tensor([text]))[0, -1].log_softmax(dim=-1)
                best = log_p.topk(width)
                candidates += [(text + [int(w)], score + float(p)) for p, w in zip(best.values, best.indices)]
            candidates.sort(key=lambda c: -c[1])
            beams = []
            for text, score in candidates[:width]:
                (done if text[-1] == eos else beams).append((text, score))
            if not beams or (done and max(s for _, s in done) >= max(s for _, s in beams)):
                break
    pool = done + beams
    return max(pool, key=lambda c: c[1])


def forward_cached(model, tokens, cache=None):
    # the network of TinyTransformer (pre-norm) on the new words only: the keys and values of the words already read are
    # taken from `cache` (a list with one pair (K, V) per block, shape (batch, heads, words, d_head)) and the new ones are added.
    # Returns the scores of the new positions and the new cache.
    offset = 0 if cache is None else cache[0][0].shape[2]
    B, t = tokens.shape
    x = model.embedding(tokens) + model.position(torch.arange(offset, offset + t))
    new_cache = []
    for i, block in enumerate(model.layers):
        attention = block.attention
        q, k, v = attention.qkv(block.ln1(x)).view(B, t, 3, attention.heads, -1).permute(2, 0, 3, 1, 4)
        if cache is not None:
            k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
        new_cache.append((k, v))
        allowed = torch.arange(k.shape[2])[None, :] <= (offset + torch.arange(t))[:, None]          # the causal mask, shifted
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + attention.out(out.transpose(1, 2).reshape(B, t, -1))
        x = x + block.mlp(block.ln2(x))
    return model.unembed(model.final(x)), new_cache


def generate_cached(model, prefix, steps, eos=2):
    # greedy writing with the cache: the prefix is read once, then one word at a time
    text = list(prefix)
    with torch.no_grad():
        scores, cache = forward_cached(model, torch.tensor([text]))
        for _ in range(steps):
            word = int(scores[0, -1].argmax())
            text.append(word)
            if word == eos:
                break
            scores, cache = forward_cached(model, torch.tensor([[word]]), cache)
    return text


def cached(path, train):
    # a trained network saved outside the repository: loaded from `path` when the file exists, otherwise trained by calling
    # `train()` and saved there. Returns (network, loaded). The file keeps the whole object, so it has to be deleted when the data,
    # the settings or this module change.
    if path is not None and os.path.exists(path):
        return torch.load(path, weights_only=False), True
    model = train()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
    return model, False


# ---- L1.5: fine-tuning a pre-trained network, and LoRA
def fine_tune(model, sequences, sep, steps=1500, lr=1e-3, seed=0, batch_size=64):
    # continues the training of `model` on `prefix <sep> continuation` texts (loss only after `sep`, as fit_continuation) and changes ONLY
    # the parameters that require a gradient: all of them for a full fine-tuning, the low-rank matrices after add_lora
    torch.manual_seed(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    start = (sequences == sep).int().argmax(dim=1)
    losses = []
    for _ in range(steps):
        i = torch.randint(len(sequences), (batch_size,), generator=generator)
        x = sequences[i]
        target = x[:, 1:].clone()
        target[torch.arange(target.shape[1])[None] < start[i][:, None]] = 0
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    model.loss_curve = np.array(losses).reshape(-1, 100).mean(axis=1) if steps % 100 == 0 else np.array(losses)
    return model


def lm_loss(model, sequences):
    # the mean next-word loss over all the words (the padding is ignored): how well the network still writes plain texts
    with torch.no_grad():
        logits = model(sequences[:, :-1])
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), sequences[:, 1:].reshape(-1), ignore_index=0).item()


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    # y = W x + (alpha / r) B A x : the matrix W of a linear layer is frozen and the change is the product of two thin matrices,
    # A (r x d_in) and B (d_out x r). B starts at zero, so at the start the layer is exactly the original one.
    def __init__(self, base, r, alpha=None):
        super().__init__()
        self.base, self.r = base, r
        self.scale = (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.A.T @ self.B.T) * self.scale if self.enabled else y

    def delta(self):
        # the change of the matrix W that this layer adds, (d_out x d_in)
        return (self.B @ self.A) * self.scale


def add_lora(model, r, where=("attention",), alpha=None, seed=0):
    # freezes every parameter and puts a LoRALinear in place of each linear layer inside the parts of the blocks named in `where`
    # ("attention": the matrices of the attention; "mlp": the two matrices of the MLP). The only trainable parameters are the A and B.
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for part in where:
            module = getattr(block, part)
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, r, alpha))
    return model


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_adapter(model, on):
    # turns the low-rank change of every LoRA layer on or off: off, the network is exactly the pre-trained one
    for layer in lora_layers(model):
        layer.enabled = on
    return model


def adapter_state(model):
    # the numbers of the adapter only (the A and B of each layer), copied: this is what has to be saved for a task
    return {name: p.detach().clone() for name, p in model.named_parameters() if name.endswith((".A", ".B"))}


def load_adapter(model, state):
    # puts the numbers of an adapter in a network that has LoRA layers of the same shape
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    return model


def merge_lora(model):
    # W <- W + (alpha / r) B A in every LoRA layer, which is replaced by a plain linear layer: same outputs, no extra computation
    model = copy.deepcopy(model)
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                merged = nn.Linear(child.base.in_features, child.base.out_features, bias=child.base.bias is not None)
                with torch.no_grad():
                    merged.weight.copy_(child.base.weight + (child.delta() if child.enabled else 0))
                    if child.base.bias is not None:
                        merged.bias.copy_(child.base.bias)
                setattr(parent, name, merged)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ---- L1.6: instruction tuning, reward model, RLHF (REINFORCE) and DPO
def sample_responses(model, prefixes, sep, max_new=22, eos=2, length=40, generator=None):
    # writes a response to each `prefix <sep>` by SAMPLING from the probabilities of the network (temperature 1), not by taking the
    # most probable word. `prefixes` is a (batch, T0) tensor whose last column is `sep`; the result is (batch, length), padded with 0
    x = prefixes
    done = torch.zeros(len(x), dtype=torch.bool)
    with torch.no_grad():
        for _ in range(max_new):
            nxt = torch.multinomial(torch.softmax(model(x)[:, -1], dim=-1), 1, generator=generator).squeeze(1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)
            done = done | (nxt == eos)
            if done.all():
                break
    return nn.functional.pad(x, (0, length - x.shape[1]))


def response_logprob(model, x, sep, eos=2):
    # log-probability that the network gives to the response of each text: the sum over the words after `sep`, `</s>` included
    logp = torch.log_softmax(model(x[:, :-1]), dim=-1).gather(2, x[:, 1:, None]).squeeze(2)
    position = torch.arange(x.shape[1] - 1)[None]
    start, end = (x == sep).int().argmax(dim=1)[:, None], (x == eos).int().argmax(dim=1)[:, None]
    return (logp * ((position >= start) & (position <= end - 1))).sum(dim=1)


class RewardModel(nn.Module):
    # a copy of a pre-trained decoder whose output layer is replaced by a single number: the score of a whole text, read in the
    # vector of its last word (`</s>`)
    def __init__(self, model, eos=2):
        super().__init__()
        self.body, self.eos = copy.deepcopy(model), eos
        self.score = nn.Linear(self.body.unembed.in_features, 1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        body = self.body
        h = body.embedding(x) + body.position(torch.arange(x.shape[1]))
        allowed = body.allowed(x)
        for block in body.blocks:
            h = block(h, allowed)
        h = body.norm(h)
        return self.score(h[torch.arange(len(x)), (x == self.eos).int().argmax(dim=1)]).squeeze(1)


def fit_reward_model(reward_model, chosen, rejected, steps=300, lr=1e-3, seed=0, batch_size=64):
    # Bradley-Terry: the probability that `chosen` beats `rejected` is sigmoid(score(chosen) - score(rejected)); we minimize -log of it
    torch.manual_seed(seed)
    for p in reward_model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        i = torch.randint(len(chosen), (batch_size,))
        loss = -nn.functional.logsigmoid(reward_model(chosen[i]) - reward_model(rejected[i])).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reward_model.eval().requires_grad_(False)


def reinforce(policy, prompts, reward, sep, beta=0.0, steps=300, lr=3e-3, seed=0, batch_size=64, callback=None, checkpoints=()):
    # RLHF in its simplest form (a simplified REINFORCE, not PPO): the policy (a network with LoRA layers whose adapter is on) writes a
    # response to each prompt, the reward is reward(text) - beta * (log p_policy - log p_reference) of the response, where the reference
    # is the SAME network with the adapter off, and the log-probability of the responses is pushed up in proportion to how much
    # their reward is above the average of the batch
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng, generator = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        x = sample_responses(policy, prompts[torch.as_tensor(rng.integers(len(prompts), size=batch_size))], sep, generator=generator)
        logp = response_logprob(policy, x, sep)
        with torch.no_grad():
            set_adapter(policy, False)
            reference = response_logprob(policy, x, sep)
            set_adapter(policy, True)
            r = reward(x) - beta * (logp.detach() - reference)
        loss = -((r - r.mean()) * logp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if callback is not None and step in checkpoints:
            callback(step, policy)
    return policy


def dpo(policy, chosen, rejected, sep, beta=0.5, steps=300, lr=1e-3, seed=0, batch_size=64, callback=None, checkpoints=()):
    # DPO: no reward model and no sampling. Loss = -log sigmoid(beta * [(log p(chosen) - log p_ref(chosen)) - (log p(rejected) - log p_ref(rejected))]);
    # the reference is again the network with the adapter off (its numbers are computed once, before the training)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    with torch.no_grad():
        set_adapter(policy, False)
        reference_chosen, reference_rejected = response_logprob(policy, chosen, sep), response_logprob(policy, rejected, sep)
        set_adapter(policy, True)
    for step in range(1, steps + 1):
        i = torch.randint(len(chosen), (batch_size,))
        margin = (response_logprob(policy, chosen[i], sep) - reference_chosen[i]) - (response_logprob(policy, rejected[i], sep) - reference_rejected[i])
        loss = -nn.functional.logsigmoid(beta * margin).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if callback is not None and step in checkpoints:
            callback(step, policy)
    return policy
