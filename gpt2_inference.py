"""GPT-2 Inference Script.

Generates text using standard autoregressive GPT-2 for baseline comparison
with RoBERTa diffusion models.
"""

import argparse
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_device

# =============================================================================
# Configuration
# =============================================================================


class Config:
    """Inference configuration."""

    MODEL_NAME: str = "gpt2"
    MAX_LENGTH: int = 256
    TOP_K: int = 50
    TOP_P: float = 0.95
    TEMPERATURE: float = 0.8


# =============================================================================
# Generation
# =============================================================================


def generate(
    prompt_text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    config: Config,
    device: torch.device,
) -> tuple[str, float]:
    """Generate text using GPT-2 autoregressive sampling.

    Args:
        prompt_text: Input text prompt
        model: GPT-2 model
        tokenizer: Tokenizer instance
        config: Configuration object
        device: Device to run on

    Returns:
        Tuple of (generated_text, elapsed_time)
    """
    # Tokenize the prompt
    print("[INFO] Tokenizing prompt…")
    encoding = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    print(f"[INFO] Prompt token length = {input_ids.shape[-1]}")

    # Generate continuation
    print("[INFO] Starting text generation…")
    t0 = time.time()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=config.MAX_LENGTH,
        do_sample=True,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        temperature=config.TEMPERATURE,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    elapsed = time.time() - t0
    print(f"[INFO] Generation took {elapsed:.2f} seconds")

    # Decode the generated text
    print("[INFO] Decoding generated tokens…")
    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return generated_text, elapsed


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main inference function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run GPT-2 inference for baseline comparison."
    )
    parser.add_argument("prompt", type=str, help="Text prompt for generation.")
    args = parser.parse_args()

    config = Config()
    device = get_device()

    # Load model and tokenizer
    print("[INFO] Loading GPT-2 tokenizer and model…")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.MODEL_NAME)
    # GPT-2 has no pad_token by default, set it to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # Generate text
    generated_text, elapsed = generate(args.prompt, model, tokenizer, config, device)

    # Display output
    print("\n" + "=" * 60)
    print("Generated Text")
    print("=" * 60)
    print(generated_text)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
