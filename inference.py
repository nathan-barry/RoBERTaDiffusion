"""RoBERTa Diffusion Inference Script.

Generates text using confidence-based parallel decoding with a trained RoBERTa diffusion model.
Supports optional matplotlib animations to visualize the generation process.
"""

import argparse
import time

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from config import Config, get_device


# =============================================================================
# Prompt Preparation
# =============================================================================


def prepare_prompt(
    prompt_text: str,
    tokenizer: RobertaTokenizerFast,
    prefix_len: int,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompt and prepare initial masked sequence.

    Args:
        prompt_text: Input text prompt
        tokenizer: Tokenizer instance
        prefix_len: Number of prefix tokens to keep fixed
        max_len: Maximum sequence length
        device: Device to place tensors on

    Returns:
        Tuple of (input_ids, attention_mask) tensors
    """
    # Tokenize the prompt
    encoding = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    input_ids_prompt = encoding["input_ids"].squeeze(0)

    # Truncate or pad the prompt to exactly PREFIX_LEN tokens
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    L_p = input_ids_prompt.size(0)

    if L_p >= prefix_len:
        context_ids = input_ids_prompt[:prefix_len].clone()
    else:
        num_left_pad = prefix_len - L_p
        pad_tensor = torch.full((num_left_pad,), fill_value=pad_id, dtype=torch.long)
        context_ids = torch.cat([pad_tensor, input_ids_prompt], dim=0)

    # Initialize with all masks
    mask_id = tokenizer.mask_token_id
    current_ids = torch.full((1, max_len), fill_value=mask_id, dtype=torch.long)

    # Place the fixed prefix
    current_ids[0, :prefix_len] = context_ids

    # Create attention mask
    current_attention = torch.ones((1, max_len), dtype=torch.long)

    return current_ids.to(device), current_attention.to(device)


# =============================================================================
# Confidence-Based Parallel Decoding
# =============================================================================


def generate(
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    initial_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: Config,
    device: torch.device,
    collect_snapshots: bool = False,
) -> tuple[torch.Tensor, list[str], float]:
    """Generate text using confidence-aware parallel decoding.

    At each step, decode all tokens whose confidence exceeds a threshold.
    This allows multiple tokens to be decoded in parallel per step.

    Args:
        model: Trained RoBERTa model
        tokenizer: Tokenizer instance
        initial_ids: Initial token IDs (prefix + masks)
        attention_mask: Attention mask
        config: Configuration object
        device: Device to run on
        collect_snapshots: Whether to collect intermediate states for animation

    Returns:
        Tuple of (final_ids, snapshots, elapsed_time)
    """
    x = initial_ids.clone()
    batch_size = x.size(0)
    seq_len = x.size(1)

    # Track which positions are still masked
    masked_positions = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    masked_positions[:, : config.PREFIX_LEN] = False  # Context is not masked

    print(
        f"[INFO] Starting confidence-based generation (threshold={config.CONFIDENCE_THRESHOLD})…"
    )

    snapshots = []
    if collect_snapshots:
        snapshots.append(
            tokenizer.decode(
                x[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )[3:]
        )

    t0 = time.time()
    step = 0

    while masked_positions.any():
        step += 1

        # Predict tokens
        with torch.no_grad():
            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs.logits

        # Get confidence scores (max probability for each position)
        probs = torch.softmax(logits / config.TEMPERATURE, dim=-1)
        confidences, predicted_tokens = torch.max(probs, dim=-1)

        # Select positions above threshold (only among masked positions)
        above_threshold = (
            confidences >= config.CONFIDENCE_THRESHOLD
        ) & masked_positions

        # Ensure at least one token is decoded per batch if any remain masked
        for b in range(batch_size):
            if masked_positions[b].any() and not above_threshold[b].any():
                # Decode the highest confidence masked token
                masked_confidences = confidences[b].clone()
                masked_confidences[~masked_positions[b]] = -float("inf")
                best_idx = torch.argmax(masked_confidences)
                above_threshold[b, best_idx] = True

        # Update positions above threshold
        x = torch.where(above_threshold, predicted_tokens, x)
        masked_positions = masked_positions & ~above_threshold

        if collect_snapshots:
            snapshots.append(
                tokenizer.decode(
                    x[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )[3:]
            )

    elapsed = time.time() - t0
    print(f"[INFO] Generation took {elapsed:.2f} seconds ({step} steps)")

    return x, snapshots, elapsed


# =============================================================================
# Animation
# =============================================================================


def create_animation(snapshots: list[str]) -> None:
    """Create and display matplotlib animation of generation process.

    Args:
        snapshots: List of text snapshots at each generation step
    """
    # Convert snapshots to display format
    all_text_snapshots = [snap.replace("<mask>", " ____") for snap in snapshots]

    # Build the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    fontdict = {"family": "monospace", "fontsize": 10}

    def update(frame_idx: int) -> None:
        """Update animation frame."""
        ax.clear()
        ax.axis("off")

        text_to_display = all_text_snapshots[frame_idx]

        ax.text(
            0.00,
            1.00,
            text_to_display,
            fontdict=fontdict,
            ha="left",
            va="top",
            wrap=True,
            transform=ax.transAxes,
        )

        ax.set_title(f"Step {frame_idx} / {len(all_text_snapshots) - 1}", pad=12)

    anim = FuncAnimation(
        fig,
        update,
        frames=range(len(all_text_snapshots)),
        interval=500,
        blit=False,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main inference function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run RoBERTa diffusion inference with optional animation."
    )
    parser.add_argument(
        "--animation",
        action="store_false",
        help="If set, skip creating or showing the animation.",
    )
    parser.add_argument(
        "prompt", type=str, help="Text prompt to use as the fixed prefix."
    )
    args = parser.parse_args()

    prompt_text = args.prompt
    animate = not args.animation

    config = Config()
    device = get_device()

    # Load model and tokenizer
    print("[INFO] Loading RoBERTa tokenizer and model…")
    tokenizer = RobertaTokenizerFast.from_pretrained(config.OUTPUT_DIR)
    model = RobertaForMaskedLM.from_pretrained(config.OUTPUT_DIR)
    model.to(device)
    model.eval()

    # Prepare prompt
    print("[INFO] Tokenizing prompt…")
    initial_ids, attention_mask = prepare_prompt(
        prompt_text, tokenizer, config.PREFIX_LEN, config.MAX_LEN, device
    )

    # Generate
    final_ids, snapshots, elapsed = generate(
        model, tokenizer, initial_ids, attention_mask, config, device, animate
    )

    # Display final output
    print("\n" + "=" * 60)
    print("Final Output")
    print("=" * 60)
    decoded = tokenizer.decode(
        final_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(decoded.replace(tokenizer.mask_token, "_____"))
    print("=" * 60 + "\n")

    # Show animation if requested
    if animate:
        create_animation(snapshots)


if __name__ == "__main__":
    main()
