import sys
import argparse
import time
import torch
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from transformers import RobertaTokenizerFast, RobertaForMaskedLM

# --------------------------------------
# 0) User Config
# --------------------------------------
MODEL_DIR = "weights/roberta-diffusion-16s40e"
MAX_LEN = 256
PREFIX_LEN = 16
N_STEPS = 10

parser = argparse.ArgumentParser(
    description="Run RoBERTa‐diffusion inference, optionally with a matplotlib animation."
)
parser.add_argument(
    "--animation",
    action="store_false",
    help="If set, skip creating or showing the animation.",
)
parser.add_argument("prompt", type=str, help="Text prompt to use as the fixed prefix.")
args = parser.parse_args()

prompt_text = args.prompt
animate = not args.animation

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("[INFO] Using MPS (Apple silicon) backend")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[INFO] Using CUDA backend")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU backend")


# =============================================================================
# Minimal top-k / top-p (nucleus) filtering function (identical to your inference.py)
# -----------------------------------------------------------------------------
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
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


# --------------------------------------
# 1) Load tokenizer & model from disk
# --------------------------------------
print("[INFO] Loading RoBERTa tokenizer and model…")
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForMaskedLM.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Capture the literal string used for the mask token (e.g. "<mask>")
mask_str = tokenizer.mask_token

# --------------------------------------
# 2) Tokenize the prompt (no padding)
# --------------------------------------
print("[INFO] Tokenizing prompt…")
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
# 7) Build the list of inference mask‐probabilities
# --------------------------------------
mask_probs = [i / N_STEPS for i in range(N_STEPS - 1, -1, -1)]
print("[INFO] Mask probabilities (high → low):", mask_probs)

# --------------------------------------
# 8) Multi‐step denoising, collecting snapshots if needed
# --------------------------------------
print("[INFO] Starting text generation…")

if animate:
    # Only collect snapshots if we’re animating
    snapshots = [current_ids[0].detach().cpu().clone()]

# Start timing right before the denoising loop
t0 = time.time()

for p_mask in mask_probs:
    # 8a) Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=current_ids,
            attention_mask=current_attention,
        )
        logits = outputs.logits  # shape: (1, MAX_LEN, vocab_size)

    # 8b) Sample predictions for each position
    pred_ids = torch.zeros((1, MAX_LEN), dtype=torch.long, device=DEVICE)
    for i in range(MAX_LEN):
        logit_vec = logits[0, i, :]
        filtered = top_k_top_p_filtering(
            logit_vec,
            top_k=50,
            top_p=0.95,
            filter_value=-float("Inf"),
        )
        probs = torch.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        pred_ids[0, i] = sampled

    # 8c) If p_mask == 0.0: reveal everything ≥ PREFIX_LEN and stop
    if p_mask == 0.0:
        new_ids = current_ids.clone()
        new_ids[0, PREFIX_LEN:] = pred_ids[0, PREFIX_LEN:]
        current_ids = new_ids
        if animate:
            snapshots.append(current_ids[0].detach().cpu().clone())
        break

    # 8d) Otherwise: randomly re-mask a fraction p_mask of tokens ≥ PREFIX_LEN
    positions = torch.arange(MAX_LEN, device=DEVICE)
    is_prefix = positions < PREFIX_LEN
    can_modify = ~is_prefix

    rand = torch.rand(MAX_LEN, device=DEVICE)
    mask_positions = (rand < p_mask) & can_modify  # True => re‐mask

    # 8e) Build next_ids
    next_ids = current_ids.clone()
    for i in range(PREFIX_LEN, MAX_LEN):
        if mask_positions[i]:
            next_ids[0, i] = mask_id
        else:
            next_ids[0, i] = pred_ids[0, i]

    current_ids = next_ids
    if animate:
        snapshots.append(current_ids[0].detach().cpu().clone())

# End timing immediately after the denoising loop
t1 = time.time()
elapsed = t1 - t0
print(f"[INFO] Denoising loop took {elapsed:.2f} seconds")


# --------------------------------------
# 9) Final textual output (no animation)
# --------------------------------------
print("\n=== Final Output ===")
decoded_tokens = tokenizer.convert_ids_to_tokens(current_ids[0].detach().cpu().tolist())
decoded = tokenizer.decode(
    current_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(decoded.replace(tokenizer.mask_token, "_____"))
print("====================\n")

# --------------------------------------
# 10) If animation was requested, build the matplotlib figure
# --------------------------------------
if animate:
    # Convert each snapshot → a concatenated, wrapped string
    all_text_snapshots = []
    for snap in snapshots:
        token_list = tokenizer.convert_ids_to_tokens(snap.tolist())
        processed_tokens = []
        for tok in token_list:
            if tok == mask_str:
                processed_tokens.append("_____")
            else:
                processed_tokens.append(tok)
        joined = "".join(processed_tokens)
        all_text_snapshots.append(joined)

    # Build the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Create a white buffer by shrinking the axes within the figure
    # Adjust these percentages to change the margin size:
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    fontdict = {
        "family": "monospace",
        "fontsize": 10,
    }

    def update(frame_idx):
        """Display the entire sequence at `frame_idx` as wrapped, left-aligned text."""
        ax.clear()
        ax.axis("off")

        text_to_display = all_text_snapshots[frame_idx]

        # Place the text inset so it isn't flush with the axes border
        ax.text(
            0.00,  # x in axes fraction (0.0 = left edge of axes)
            1.00,  # y in axes fraction (1.0 = top of axes)
            text_to_display,
            fontdict=fontdict,
            ha="left",
            va="top",
            wrap=True,
            transform=ax.transAxes,
        )

        ax.set_title(f"Step {frame_idx} / {len(all_text_snapshots) - 1}", pad=12)

    # Create and show the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=range(len(all_text_snapshots)),
        interval=500,  # milliseconds between frames
        blit=False,
    )

    plt.tight_layout()
    plt.show()
