"""Compare RoBERTa Diffusion and GPT-2 Generation.

Side-by-side comparison of RoBERTa confidence-based diffusion decoding
versus standard GPT-2 autoregressive generation with synchronized animation.
"""

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
)

# Import generation functions from existing inference scripts
import gpt2_inference
import inference
from config import Config, get_device


# =============================================================================
# Animation
# =============================================================================


def animate_comparison(
    roberta_snaps: list[str],
    rd_time: float,
    gpt_snaps: list[str],
    gt_time: float,
    mask_token: str,
    fps: int = 30,
) -> None:
    """Create synchronized animation comparing both models.

    Args:
        roberta_snaps: RoBERTa generation snapshots
        rd_time: RoBERTa generation time
        gpt_snaps: GPT-2 generation snapshots
        gt_time: GPT-2 generation time
        mask_token: Mask token string to replace in display
        fps: Frames per second for animation
    """
    total = max(rd_time, gt_time)
    frames = int(total * fps) + 90

    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("white")
    for ax in (ax_r, ax_g):
        ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05, hspace=0.10)

    font = {"family": "monospace", "fontsize": 10}

    def update(frame: int) -> None:
        """Update animation frame."""
        t = frame / fps

        # RoBERTa index
        i_r = min(int(t / rd_time * (len(roberta_snaps) - 1)), len(roberta_snaps) - 1)

        # GPT-2: hold prompt for first 30 frames, then map remaining frames
        if frame < 30:
            i_g = 0
        else:
            t_g = (frame - 30) / fps
            i_g = min(int(t_g / gt_time * (len(gpt_snaps) - 1)), len(gpt_snaps) - 1)

        # RoBERTa panel
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
        ax_r.set_title(f"RoBERTa Diffusion: Step {i_r}/{len(roberta_snaps) - 1}", pad=8)

        # GPT-2 panel
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
        ax_g.set_title(f"GPT-2 Generation: Token {i_g}/{len(gpt_snaps) - 1}", pad=8)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    plt.show()


# =============================================================================
# GPT-2 Snapshots Builder
# =============================================================================


def build_gpt2_snapshots(
    generated_text: str,
    tokenizer: GPT2TokenizerFast,
    prefix_len: int,
) -> list[str]:
    """Build incremental snapshots from GPT-2 generated text.

    Args:
        generated_text: Full generated text from GPT-2
        tokenizer: GPT-2 tokenizer
        prefix_len: Number of prefix tokens

    Returns:
        List of progressive text snapshots
    """
    # Tokenize the full output
    full_ids = tokenizer.encode(generated_text)
    L = len(full_ids)
    init_len = min(prefix_len, L)

    snapshots = []
    # First frame = the first PREFIX_LEN tokens
    snapshots.append(
        tokenizer.decode(
            full_ids[:init_len],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )
    # Then one more token at a time
    for i in range(init_len + 1, L + 1):
        snapshots.append(
            tokenizer.decode(
                full_ids[:i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

    return snapshots


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main comparison function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare RoBERTa diffusion & GPT-2")
    parser.add_argument("prompt", help="Prompt text for both models")
    parser.add_argument(
        "--roberta-dir",
        default="weights",
        help="Path to RoBERTa model directory",
    )
    args = parser.parse_args()

    config = Config()
    device = get_device()

    # Override model directory if specified
    model_dir = args.roberta_dir if args.roberta_dir != "weights" else config.OUTPUT_DIR

    # Load RoBERTa
    print("[INFO] Loading RoBERTa model…")
    rtok = RobertaTokenizerFast.from_pretrained(model_dir)
    rmodel = RobertaForMaskedLM.from_pretrained(model_dir)
    rmodel.to(device)
    rmodel.eval()

    # Load GPT-2
    print("[INFO] Loading GPT-2 model…")
    gtok = GPT2TokenizerFast.from_pretrained(config.GPT_MODEL_NAME)
    gtok.pad_token = gtok.eos_token
    gmodel = GPT2LMHeadModel.from_pretrained(config.GPT_MODEL_NAME)
    gmodel.to(device)
    gmodel.eval()

    # RoBERTa generation (reuse from inference.py)
    print("[INFO] Running RoBERTa diffusion…")
    initial_ids, attention_mask = inference.prepare_prompt(
        args.prompt, rtok, config.PREFIX_LEN, config.MAX_LEN, device
    )
    rd_final, rd_snaps, rd_time = inference.generate(
        rmodel,
        rtok,
        initial_ids,
        attention_mask,
        config,
        device,
        collect_snapshots=True,
    )
    print(f"[RESULT] RoBERTa: {rd_time:.2f}s over {len(rd_snaps)} steps")

    # GPT-2 generation (reuse from gpt2_inference.py)
    print("[INFO] Running GPT-2 generation…")
    gt_text, gt_time = gpt2_inference.generate(
        args.prompt,
        gmodel,
        gtok,
        config,
        device,
    )
    print(f"[RESULT] GPT-2: {gt_time:.2f}s")

    # Build GPT-2 snapshots for animation
    gt_snaps = build_gpt2_snapshots(gt_text, gtok, config.PREFIX_LEN)

    # Display comparison animation
    animate_comparison(
        rd_snaps, rd_time, gt_snaps, gt_time, rtok.mask_token, config.ANIMATION_FPS
    )


if __name__ == "__main__":
    main()
