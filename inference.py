import sys
import torch
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from transformers import RobertaTokenizerFast, RobertaForMaskedLM

# === USER CONFIG ===
MODEL_DIR = "weights/roberta-diffusion-single-with-prefix"
MAX_LEN = 256
PREFIX_LEN = 16
N_STEPS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Minimal top-k / top-p (nucleus) filtering function (identical to your inference.py)
# -----------------------------------------------------------------------------
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: 1D tensor of shape (vocab_size,)
        top_k: keep only top_k tokens with highest logits (set others to filter_value)
        top_p: keep the smallest set of tokens whose cumulative probability ≥ top_p
        filter_value: the value to assign to filtered logits (default: -inf)

    Returns:
        The filtered logits (same shape, with unwanted positions = filter_value).

    """
    logits = logits.clone()

    # ===== Top-K filtering =====
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1]
        indices_to_remove = logits < kth_value
        logits[indices_to_remove] = filter_value

    # ===== Top-P (nucleus) filtering =====
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


if len(sys.argv) < 2:
    print('Usage: python inference.py "<PROMPT>"')
    sys.exit(1)

prompt_text = sys.argv[1]

# --------------------------------------
# 1) Load tokenizer & model from disk
# --------------------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForMaskedLM.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Capture the literal string used for the mask token (e.g. "<mask>")
mask_str = tokenizer.mask_token

# --------------------------------------
# 2) Tokenize the prompt (no padding)
# --------------------------------------
encoding = tokenizer(
    prompt_text,
    truncation=False,
    padding=False,
    return_tensors="pt",
)
input_ids_prompt = encoding["input_ids"].squeeze(0)  # shape: (L_p,)

# --------------------------------------
# 3) Truncate or pad the prompt to exactly PREFIX_LEN tokens
# --------------------------------------
pad_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)
L_p = input_ids_prompt.size(0)

if L_p >= PREFIX_LEN:
    # Truncate to first PREFIX_LEN tokens
    context_ids = input_ids_prompt[:PREFIX_LEN].clone()
else:
    # Left-pad with <pad> so length == PREFIX_LEN
    num_left_pad = PREFIX_LEN - L_p
    pad_tensor = torch.full((num_left_pad,), fill_value=pad_id, dtype=torch.long)
    context_ids = torch.cat([pad_tensor, input_ids_prompt], dim=0)

# --------------------------------------
# 4) Initialize "current_ids" = a (1 x MAX_LEN) tensor of <mask>
# --------------------------------------
mask_id = tokenizer.mask_token_id
current_ids = torch.full((1, MAX_LEN), fill_value=mask_id, dtype=torch.long)

# --------------------------------------
# 5) Place the fixed prefix (context_ids) into positions [0..PREFIX_LEN-1]
# --------------------------------------
current_ids[0, :PREFIX_LEN] = context_ids

# --------------------------------------
# 6) attention_mask = all 1's
# --------------------------------------
current_attention = torch.ones((1, MAX_LEN), dtype=torch.long)

current_ids = current_ids.to(DEVICE)
current_attention = current_attention.to(DEVICE)

# --------------------------------------
# 7) Build the list of inference mask-probabilities
# --------------------------------------
mask_probs = [i / N_STEPS for i in range(N_STEPS - 1, -1, -1)]
print(mask_probs)

# --------------------------------------
# 8) Multi-step denoising, but first collect a list of snapshots for animation.
#    Each snapshot is the "current_ids" after updating for that step.
#    We explicitly skip appending a snapshot for p_mask == 1.0, so that the first
#    “revealed” frame occurs at p_mask = 0.8.
# --------------------------------------
snapshots = [current_ids[0].detach().cpu().clone()]  # start with all-masks

for p_mask in mask_probs:
    # 8a) Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=current_ids,
            attention_mask=current_attention,
        )
        logits = outputs.logits  # shape: (1, MAX_LEN, vocab_size)

    # 8b) Sample predictions for all positions
    pred_ids = torch.zeros((1, MAX_LEN), dtype=torch.long, device=DEVICE)
    for i in range(MAX_LEN):
        logit_vec = logits[0, i, :]  # shape (vocab_size,)
        filtered = top_k_top_p_filtering(
            logit_vec,
            top_k=50,
            top_p=0.95,
            filter_value=-float("Inf"),
        )
        probs = torch.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        pred_ids[0, i] = sampled

    # 8c) If p_mask == 0.0, fill everything ≥ PREFIX_LEN with pred_ids and break.
    if p_mask == 0.0:
        new_ids = current_ids.clone()
        new_ids[0, PREFIX_LEN:] = pred_ids[0, PREFIX_LEN:]
        current_ids = new_ids
        snapshots.append(current_ids[0].detach().cpu().clone())
        break

    # 8d) Otherwise, randomly re-mask a fraction p_mask of positions ≥ PREFIX_LEN
    positions = torch.arange(MAX_LEN, device=DEVICE)
    is_prefix = positions < PREFIX_LEN
    can_modify = ~is_prefix

    # Randomly decide which positions to re-mask
    rand = torch.rand(MAX_LEN, device=DEVICE)
    mask_positions = (rand < p_mask) & can_modify  # True => will be masked

    # 8e) Build the next current_ids:
    next_ids = current_ids.clone()
    for i in range(PREFIX_LEN, MAX_LEN):
        if mask_positions[i]:
            next_ids[0, i] = mask_id
        else:
            next_ids[0, i] = pred_ids[0, i]

    current_ids = next_ids

    snapshots.append(current_ids[0].detach().cpu().clone())

snapshots.append(current_ids[0].detach().cpu().clone())
snapshots.append(current_ids[0].detach().cpu().clone())

# --------------------------------------
# 9) Convert each snapshot of IDs → a list of token strings
# --------------------------------------
all_token_grids = []  # list of lists-of-strings, shape: [n_steps][MAX_LEN]
for snap in snapshots:
    ids = snap.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    all_token_grids.append(tokens)

# --------------------------------------
# *** PRINT FINAL OUTPUT TO CONSOLE ***
# --------------------------------------
# The last element in `snapshots` is the fully denoised sequence.
final_ids = snapshots[-1].tolist()
final_str = tokenizer.decode(
    final_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print("\n=== Final Output ===")
print(final_str)
print("====================\n")

# --------------------------------------
# 10) Prepare for animation: compute grid layout
# --------------------------------------
seq_len = MAX_LEN
num_cols = math.ceil(math.sqrt(seq_len))
num_rows = math.ceil(seq_len / num_cols)

xs, ys = [], []
x_margin = 0.02
y_margin = 0.02
usable_width = 1.0 - 2 * x_margin
usable_height = 1.0 - 2 * y_margin

for row in range(num_rows):
    y_center = 1.0 - (y_margin + (row + 0.5) * (usable_height / num_rows))
    for col in range(num_cols):
        x_center = x_margin + (col + 0.5) * (usable_width / num_cols)
        xs.append(x_center)
        ys.append(y_center)
xs = xs[:seq_len]
ys = ys[:seq_len]

# --------------------------------------
# 11) Build Matplotlib figure for animation
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis("off")

fontdict = {
    "family": "monospace",
    "fontsize": 8,
}


def update(frame_idx):
    """At each frame, show the token strings from that snapshot in a grid.
    If a token == mask_str, display "____" instead.
    Otherwise, strip off any leading 'Ġ'.
    """
    ax.clear()
    ax.axis("off")

    tokens_this_step = all_token_grids[frame_idx]  # length = MAX_LEN
    for i, tok in enumerate(tokens_this_step):
        if tok == mask_str:
            display_tok = "____"
        else:
            # Strip leading 'Ġ' if present
            display_tok = tok[1:] if tok.startswith("Ġ") else tok

        ax.text(
            xs[i],
            ys[i],
            display_tok,
            fontdict=fontdict,
            ha="center",
            va="center",
        )

    ax.set_title(f"Step {frame_idx} / {len(all_token_grids) - 1}", pad=20)


# --------------------------------------
# 12) Create and show the animation
# --------------------------------------
anim = FuncAnimation(
    fig,
    update,
    frames=range(len(all_token_grids)),
    interval=500,  # 1000 ms between frames
    blit=False,
)

plt.tight_layout()
plt.show()
