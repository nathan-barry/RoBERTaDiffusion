# inference.py

import sys
import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

# === USER CONFIG ===
MODEL_DIR = "weights/roberta-diffusion-single-with-prefix"
MAX_LEN = 256
PREFIX_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Minimal top-k / top-p (nucleus) filtering function.
# -----------------------------------------------------------------------------
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
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
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


# =============================================================================


def main():
    if len(sys.argv) < 2:
        print('Usage: python inference.py "<your prompt text (any length)>"')
        sys.exit(1)

    prompt_text = sys.argv[1]

    # 1) Load tokenizer & model from disk
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    model = RobertaForMaskedLM.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    # 2) Tokenize the prompt (no padding) to get raw token IDs
    encoding = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    input_ids_prompt = encoding["input_ids"].squeeze(0)  # shape: (L_p,)

    # 3) Truncate or pad the prompt to exactly PREFIX_LEN tokens
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    L_p = input_ids_prompt.size(0)

    if L_p >= PREFIX_LEN:
        # Truncate to first PREFIX_LEN tokens
        context_ids = input_ids_prompt[:PREFIX_LEN]
    else:
        # Left-pad with <pad> so length == PREFIX_LEN
        num_left_pad = PREFIX_LEN - L_p
        pad_tensor = torch.full((num_left_pad,), fill_value=pad_id, dtype=torch.long)
        context_ids = torch.cat([pad_tensor, input_ids_prompt], dim=0)

    # 4) Initialize “current_ids” = a (1 x MAX_LEN) tensor of <mask>
    mask_id = tokenizer.mask_token_id
    current_ids = torch.full((1, MAX_LEN), fill_value=mask_id, dtype=torch.long)

    # 5) Place the fixed prefix (context_ids) into positions [0..PREFIX_LEN-1]
    current_ids[0, :PREFIX_LEN] = context_ids

    # 6) attention_mask = all 1’s (we never used padding in training)
    current_attention = torch.ones((1, MAX_LEN), dtype=torch.long)

    current_ids = current_ids.to(DEVICE)
    current_attention = current_attention.to(DEVICE)

    # 7) Build the list of inference mask‐probabilities:
    #    We trained on mask_probs = [0.2, 0.4, 0.6, 0.8, 1.0].
    #    For inference, we go: 1.0 → 0.8 → 0.6 → 0.4 → 0.2 → 0.0
    mask_probs = [(i + 1) / 5 for i in range(5)]  # [0.2, 0.4, 0.6, 0.8, 1.0]
    inference_probs = list(reversed(mask_probs)) + [
        0.0
    ]  # [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    # 8) Multi‐step denoising:
    #    After each step, we will print “Masked Input” and “Output” using '_' instead of '?'
    for step, p_mask in enumerate(inference_probs):
        # 8a) Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_attention,
            )
            logits = outputs.logits  # shape: (1, MAX_LEN, vocab_size)

        # 8b)predictions (shape: (1, MAX_LEN))
        pred_ids = torch.zeros((1, MAX_LEN), dtype=torch.long, device=DEVICE)
        for i in range(MAX_LEN):
            # skip prefix or unmasked positions if desired
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

        # 8c) Prepare strings for printing:
        #     - masked_str_step: show what the model saw (replace <mask>→'�')
        masked_str_step = tokenizer.decode(
            current_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).replace(tokenizer.mask_token, "�")

        #     - output_str_step: show the model’s pred_ids (skip special tokens)
        output_str_step = tokenizer.decode(
            pred_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 8d) Print for this step
        print(f"\n=== Step {step} (p_mask = {p_mask:.2f}) ===")
        print("Masked Input Seen by Model:")
        print(masked_str_step)
        print("\nOutput:")
        print(output_str_step)

        # 8e) If p_mask == 0.0, we keep everything ≥ PREFIX_LEN and break
        if p_mask == 0.0:
            current_ids[0, PREFIX_LEN:] = pred_ids[0, PREFIX_LEN:]
            break

        # 8f) Otherwise, randomly re-mask a fraction p_mask of positions ≥ PREFIX_LEN
        batch_input = current_ids[0]  # shape: (MAX_LEN,)
        special_ids = set(tokenizer.all_special_ids)

        # Build boolean masks for positions that CANNOT be modified (prefix or special)
        is_special = torch.zeros_like(batch_input, dtype=torch.bool)
        for sid in special_ids:
            is_special |= batch_input == sid

        positions = torch.arange(MAX_LEN, device=DEVICE)
        is_prefix = positions < PREFIX_LEN
        can_modify = (~is_prefix) & (~is_special)

        # Randomly decide which positions to re-mask
        rand = torch.rand(MAX_LEN, device=DEVICE)
        mask_positions = (rand < p_mask) & can_modify

        # 8g) Build the next current_ids:
        next_ids = current_ids.clone()
        for i in range(PREFIX_LEN, MAX_LEN):
            if mask_positions[i]:
                next_ids[0, i] = mask_id
            else:
                next_ids[0, i] = pred_ids[0, i]

        current_ids = next_ids

    # 9) Final summary printout
    print(f"\n(Completed over {len(inference_probs)} steps.)\n")


if __name__ == "__main__":
    main()
