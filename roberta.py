import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import trange

from gpt import (
    batch_size,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    device,
    estimate_loss,
    Head,
    MultiHeadAttention,
    FeedForward,
    Block,
)

# RoBERTa‐specific hyperparams
mask_prob = 0.15  # fraction of tokens to mask
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

torch.manual_seed(1337)
# ------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# add <mask> token
mask_token = "<mask>"
mask_token_id = vocab_size
stoi[mask_token] = mask_token_id
itos[mask_token_id] = mask_token
vocab_size += 1
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def mask_batch(x):
    labels = x.clone()
    prob = torch.rand(x.shape, device=device)
    mask = prob < mask_prob
    labels[~mask] = -100
    x = x.clone()
    x[mask] = mask_token_id
    return x, labels


def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
    return mask_batch(x)


# Switch out Head.forward from causal masking to bidirectional
def _head_forward_no_mask(self, x):
    B, T, C = x.shape
    k = self.key(x)  # (B,T,hs)
    q = self.query(x)  # (B,T,hs)
    # full bidirectional affinities
    wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,T)
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)  # (B,T,hs)
    return wei @ v  # (B,T,hs)


Head.forward = _head_forward_no_mask


# --- RoBERTa model definition reusing Block, etc. ---
class RoBERTaForMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx, labels=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device).unsqueeze(0))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), labels.view(B * T), ignore_index=-100
            )
        return logits, loss


# --- training loop ---
model = RoBERTaForMaskedLM().to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = trange(max_iters, desc="RoBERTa Training", unit="step")
for iter in pbar:
    # eval every eval_interval steps
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        pbar.set_postfix(train=f"{losses['train']:.4f}", val=f"{losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "roberta_weights.pt")
print("RoBERTa weights saved to roberta_weights.pt")
