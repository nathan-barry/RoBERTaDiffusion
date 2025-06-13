#!/usr/bin/env python3
import argparse
import time
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)


# --------------------------------------
# 0) top-k / top-p filtering
# --------------------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k)[0][..., -1]
        logits[logits < kth] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        remove_idx = sorted_idx[mask]
        logits[remove_idx] = filter_value
    return logits


# --------------------------------------
# 1) RoBERTa diffusion (exact prefix preservation)
# --------------------------------------
def run_roberta_diffusion(
    model, tokenizer, prompt, max_len, prefix_len, n_steps, device
):
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # tokenize + left-pad/truncate to prefix_len
    enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    seq = enc.input_ids.squeeze(0)
    if seq.size(0) >= prefix_len:
        ctx = seq[:prefix_len]
    else:
        pad = torch.full((prefix_len - seq.size(0),), pad_id, dtype=torch.long)
        ctx = torch.cat([pad, seq], dim=0)

    # initialize all-masks, then insert prefix
    current = torch.full((1, max_len), mask_id, dtype=torch.long)
    current[0, :prefix_len] = ctx
    current = current.to(device)
    attention = torch.ones_like(current, device=device)

    # build mask-probs high → low
    mask_probs = [i / n_steps for i in range(n_steps - 1, -1, -1)]
    snapshots = [
        tokenizer.decode(
            current[0], skip_special_tokens=False, clean_up_tokenization_spaces=True
        )[3:]
    ]

    t0 = time.time()
    positions = torch.arange(max_len, device=device)
    for p in mask_probs:
        # 1) forward
        with torch.no_grad():
            logits = model(input_ids=current, attention_mask=attention).logits

        # 2) sample preds for every position
        preds = torch.zeros_like(current)
        for i in range(max_len):
            vec = logits[0, i]
            filt = top_k_top_p_filtering(vec, top_k=50, top_p=0.95)
            probs = torch.softmax(filt, dim=-1)
            preds[0, i] = torch.multinomial(probs, 1)

        # 3) final reveal if p==0
        if math.isclose(p, 0.0):
            current[0, prefix_len:] = preds[0, prefix_len:]
            snapshots.append(
                tokenizer.decode(
                    current[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )[3:]
            )
            break

        # 4) otherwise, randomly re-mask exactly p fraction _only_ for i>=prefix_len
        rand = torch.rand(max_len, device=device)
        can_modify = positions >= prefix_len
        mask_positions = (rand < p) & can_modify

        next_ids = current.clone()
        # mask those positions
        next_ids[0, mask_positions] = mask_id
        # fill others (≥prefix_len) from preds
        keep_pred = (~mask_positions) & can_modify
        next_ids[0, keep_pred] = preds[0, keep_pred]
        # prefix (can_modify=False) stays from current.clone()
        current = next_ids

        snapshots.append(
            tokenizer.decode(
                current[0], skip_special_tokens=False, clean_up_tokenization_spaces=True
            )[3:]
        )

    rd_time = time.time() - t0
    return snapshots, rd_time


# --------------------------------------
# 2) GPT-2 generate + reveal words over time
# --------------------------------------
def run_gpt2_generate(model, tokenizer, prompt, max_len, device, prefix_len=16):
    # 1) Encode prompt
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    # 2) Generate full sequence
    t0 = time.time()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gt_time = time.time() - t0

    # 3) Build token-level snapshots starting with the first prefix_len tokens
    full_ids = output_ids[0]  # shape (L,)
    L = full_ids.size(0)
    # ensure we don’t exceed length
    init_len = min(prefix_len, L)
    snapshots = []
    # first frame = the first `prefix_len` tokens
    snapshots.append(
        tokenizer.decode(
            full_ids[:init_len],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )
    # then one more token at a time
    for i in range(init_len + 1, L + 1):
        snapshots.append(
            tokenizer.decode(
                full_ids[:i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

    return snapshots, gt_time


# --------------------------------------
# 3) Joint animation
# --------------------------------------
def animate_both(roberta_snaps, rd_time, gpt_snaps, gt_time, mask_token):
    total = max(rd_time, gt_time)
    fps = 30
    frames = int(total * fps) + 90

    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("white")
    for ax in (ax_r, ax_g):
        ax.axis("off")
    plt.subplots_adjust(
        left=0.05,
        right=0.90,
        top=0.95,
        bottom=0.05,
        hspace=0.10,
    )

    font = {"family": "monospace", "fontsize": 10}

    def update(frame):
        t = frame / fps
        # RoBERTa index (unchanged)
        i_r = min(int(t / rd_time * (len(roberta_snaps) - 1)), len(roberta_snaps) - 1)

        # GPT-2: hold prompt (index 0) for first 30 frames, then map remaining frames
        if frame < 30:
            i_g = 0
        else:
            t_g = (frame - 30) / fps
            i_g = min(int(t_g / gt_time * (len(gpt_snaps) - 1)), len(gpt_snaps) - 1)

        ax_r.clear()
        ax_r.axis("off")
        txt_r = roberta_snaps[i_r].replace(mask_token, " ____")
        ax_r.text(
            0.00,
            1.00,
            txt_r,
            fontdict=font,
            ha="left",
            va="top",
            wrap=True,
            transform=ax_r.transAxes,
        )
        ax_r.set_title(
            f"RoBERTa Diffusion:  Step {i_r}/{len(roberta_snaps) - 1}", pad=8
        )

        ax_g.clear()
        ax_g.axis("off")
        ax_g.text(
            0.00,
            1.00,
            gpt_snaps[i_g],
            fontdict=font,
            ha="left",
            va="top",
            wrap=True,
            transform=ax_g.transAxes,
        )
        ax_g.set_title(f"GPT-2 Generation:  Token {i_g}/{len(gpt_snaps) - 1}", pad=8)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    plt.show()


# --------------------------------------
# 4) Main
# --------------------------------------
def main():
    p = argparse.ArgumentParser("Compare RoBERTa‐diffusion & GPT-2")
    p.add_argument("prompt", help="Prompt text for both models")
    p.add_argument("--roberta-dir", default="weights/roberta-diffusion-16s40e")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--prefix-len", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=10)
    args = p.parse_args()

    # device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("[INFO] MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] CUDA backend")
    else:
        device = torch.device("cpu")
        print("[INFO] CPU backend")

    # load RoBERTa
    print("[INFO] Loading RoBERTa…")
    rtok = RobertaTokenizerFast.from_pretrained(args.roberta_dir)
    rmodel = RobertaForMaskedLM.from_pretrained(args.roberta_dir).to(device).eval()

    # load GPT-2
    print("[INFO] Loading GPT-2…")
    gtok = GPT2TokenizerFast.from_pretrained("gpt2")
    gtok.pad_token = gtok.eos_token
    gmodel = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    # RoBERTa diffusion
    print("[INFO] RoBERTa diffusion…")
    rd_snaps, rd_time = run_roberta_diffusion(
        rmodel, rtok, args.prompt, args.max_len, args.prefix_len, args.n_steps, device
    )
    print(f"[RESULT] RoBERTa took {rd_time:.2f}s over {len(rd_snaps)} steps")

    # GPT-2 generate
    print("[INFO] GPT-2 generation…")
    gt_snaps, gt_time = run_gpt2_generate(
        gmodel, gtok, args.prompt, args.max_len, device
    )
    print(f"[RESULT] GPT-2 took {gt_time:.2f}s over {len(gt_snaps)} words")

    # joint animation
    animate_both(rd_snaps, rd_time, gt_snaps, gt_time, rtok.mask_token)


if __name__ == "__main__":
    main()
