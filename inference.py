import os
import re
import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration: change these if needed
# ──────────────────────────────────────────────────────────────────────────────

# Directory prefix for your fine-tuned checkpoints
CHECKPOINT_DIR_PREFIX = "roberta-diffusion-"

# The max length used during training / tokenization
MAX_LEN = 256

# Example text to denoise. You can swap this out for any sentence.
# Note: we do NOT include "<mask>" manually—this script fully masks all tokens.
INPUT_TEXT = "In a distant future, humanity has spread to the stars and seeks peace."

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Find and sort all checkpoint directories by mask probability (descending)
# ──────────────────────────────────────────────────────────────────────────────


def find_and_sort_checkpoints(root_dir="."):
    """
    Finds all subdirectories in root_dir that start with CHECKPOINT_DIR_PREFIX,
    extracts the mask_prob from their name (e.g. "roberta-diffusion-03-0.33"),
    and returns a list of (mask_prob, full_path) sorted by mask_prob descending.
    """
    ckpts = []
    pattern = re.compile(
        rf"^{re.escape(CHECKPOINT_DIR_PREFIX)}\d{{2}}-([0-9]+\.[0-9]+)$"
    )
    for name in os.listdir(root_dir):
        if os.path.isdir(name):
            m = pattern.match(name)
            if m:
                prob = float(m.group(1))
                ckpts.append((prob, os.path.join(root_dir, name)))
    # sort descending by mask_prob
    ckpts.sort(key=lambda x: x[0], reverse=True)
    return ckpts


checkpoints = find_and_sort_checkpoints()
if not checkpoints:
    raise RuntimeError(f"No directories found with prefix '{CHECKPOINT_DIR_PREFIX}'")

print("Found the following checkpoints (mask_prob descending):")
for prob, path in checkpoints:
    print(f"  {path}  (mask_prob={prob:.2f})")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load a single tokenizer (all models share the same vocab/merges)
# ──────────────────────────────────────────────────────────────────────────────

# Use the tokenizer from the first checkpoint
first_ckpt_dir = checkpoints[0][1]
tokenizer = RobertaTokenizerFast.from_pretrained(first_ckpt_dir)

mask_token_id = tokenizer.mask_token_id
special_token_ids = set(tokenizer.all_special_ids)
pad_token_id = tokenizer.pad_token_id

# ──────────────────────────────────────────────────────────────────────────────
# 4) Prepare the fully-masked input IDs (x_T)
# ──────────────────────────────────────────────────────────────────────────────

# Tokenize the raw input text to exactly MAX_LEN with padding/truncation
encoded = tokenizer(
    INPUT_TEXT,
    return_tensors="pt",
    max_length=MAX_LEN,
    truncation=True,
    padding="max_length",
)
input_ids = encoded["input_ids"].to(DEVICE)  # shape: (1, MAX_LEN)

# Create a fully-masked version: replace every non-special token with <mask>
x = input_ids.clone()
# Build a boolean mask: True wherever we should mask (i.e. not a special token)
mask_positions = ~torch.isin(x, torch.tensor(list(special_token_ids), device=DEVICE))
x[mask_positions] = mask_token_id
current_ids = x  # this is x_T

print("=== Initial Fully-Masked Input (step T) ===")
print(
    tokenizer.decode(
        current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
)
print()

# ──────────────────────────────────────────────────────────────────────────────
# 5) Run the reverse diffusion: for each checkpoint in descending order
# ──────────────────────────────────────────────────────────────────────────────

for step, (prob, ckpt_dir) in enumerate(checkpoints):
    print(f"--- Step {step} (mask_prob={prob:.2f}) ---")
    # Load model weights
    model = RobertaForMaskedLM.from_pretrained(ckpt_dir).to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(current_ids)  # logits shape: (1, MAX_LEN, V)
        logits = outputs.logits
        # For each position, pick the highest-probability token
        preds = torch.argmax(logits, dim=-1)  # shape: (1, MAX_LEN)
        current_ids = preds

    # Decode for display: show � for any residual mask tokens
    decoded = tokenizer.decode(
        current_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # Replace every "<mask>" token with a � symbol
    display = decoded.replace(tokenizer.mask_token, "�")
    print(display)
    print()

print("Reverse diffusion complete. Final output:")
print(
    tokenizer.decode(
        current_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
)
print()
