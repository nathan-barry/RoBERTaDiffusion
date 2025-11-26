"""RoBERTa Diffusion Fine-tuning Script.

This script fine-tunes RoBERTa for diffusion-based text generation using a custom
masking strategy that progressively denoises text. The model learns to predict
masked tokens while preserving a fixed prefix.

Key features:
- Variable masking probabilities sampled per batch
- Fixed prefix region (first PREFIX_LEN tokens never masked)
- No special tokens masked during training
- Trained on OpenWebText dataset
"""

from typing import Any
import time

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# Configuration
# =============================================================================


class Config:
    """Training hyperparameters."""

    N_STEPS: int = 256
    BATCH_SIZE: int = 16
    MAX_LEN: int = 256
    PREFIX_LEN: int = 64
    MODEL_NAME: str = "roberta-base"
    OUTPUT_DIR: str = "weights"
    MAX_STEPS: int = 1000
    SAVE_STEPS: int = 500
    LOGGING_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 1


# =============================================================================
# Custom Data Collator
# =============================================================================


class DiffusionCollator:
    """Custom data collator for diffusion training.

    Handles tokenization and masking on-the-fly:
    1. Tokenizes raw text to exactly MAX_LEN tokens
    2. Samples a masking probability p from linearly spaced values
    3. Never masks tokens in the prefix region [0, PREFIX_LEN)
    4. Never masks special tokens (CLS, SEP, PAD, etc.)
    5. Masks proportion p of remaining valid tokens
    6. Sets unmasked tokens to -100 in labels (ignored by loss)
    """

    def __init__(self, tokenizer: RobertaTokenizerFast, config: Config) -> None:
        """Initialize the collator with tokenizer and config."""
        self.tokenizer = tokenizer
        self.config = config
        # Generate linearly spaced mask probabilities from high to low
        self.mask_probs = [
            (i + 1) / config.N_STEPS for i in range(config.N_STEPS - 1, -1, -1)
        ]
        self.special_ids = set(tokenizer.all_special_ids)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of features with tokenization and dynamic masking."""
        # Extract text from features
        # Handle both dict-like and example-like features
        texts = []
        for f in features:
            if isinstance(f, dict):
                texts.append(f.get("text", f.get("content", "")))
            else:
                # IterableDataset items might be Example objects
                texts.append(f["text"] if "text" in f else str(f))

        # Tokenize all texts to exactly MAX_LEN
        encoded = self.tokenizer(
            texts,
            max_length=self.config.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        batch_input_ids = encoded["input_ids"]
        batch_attention = encoded["attention_mask"]

        # Clone input_ids to create labels
        labels = batch_input_ids.clone()

        # Sample masking probability for this batch
        p = float(self.mask_probs[torch.randint(0, len(self.mask_probs), (1,))])

        B, L = batch_input_ids.shape

        # Identify special tokens
        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in self.special_ids:
            is_special |= batch_input_ids == sid

        # Identify prefix positions
        pos_idxs = torch.arange(L).unsqueeze(0).expand(B, L)
        is_prefix = pos_idxs < self.config.PREFIX_LEN

        # Determine which tokens can be masked
        mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)

        # Randomly select tokens to mask based on probability p
        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        mask_positions = (rand < p) & mask_candidate

        # Apply masking
        batch_input_ids[mask_positions] = self.tokenizer.mask_token_id

        # Set unmasked positions to -100 in labels (ignored by loss)
        labels[~mask_positions] = -100

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
        }


# =============================================================================
# Inference Test
# =============================================================================


def run_inference_test(
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    dataset: DatasetDict,
    diffusion_collator: DiffusionCollator,
) -> None:
    """Run a quick inference test on a sample."""
    print("\n[INFO] Running inference test...")
    model.eval()

    # Apply collator (tokenizes and masks)
    # For streaming datasets, iterate to get first sample
    sample = next(iter(dataset["train"]))
    batch = diffusion_collator([sample])
    input_ids_masked = batch["input_ids"].to(model.device)

    # Run inference
    with torch.no_grad():
        logits = model(input_ids_masked).logits
        pred_ids = logits.argmax(dim=-1)

    # Decode masked input and predictions
    masked_str = tokenizer.decode(
        input_ids_masked[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace(tokenizer.mask_token, "█")

    pred_str = tokenizer.decode(
        pred_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 60)
    print("Sample Inference (random mask rate, prefix unmasked)")
    print("=" * 60)
    print("\nMasked Input:")
    print(masked_str)
    print("\nModel Output:")
    print(pred_str)
    print("=" * 60 + "\n")


# =============================================================================
# Main Training Pipeline
# =============================================================================


def main() -> None:
    """Main training function."""
    config = Config()

    # Initialize tokenizer
    print(f"[INFO] Loading tokenizer from {config.MODEL_NAME}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(config.MODEL_NAME)
    tokenizer.model_max_length = config.MAX_LEN

    # Load dataset (no preprocessing - done on-the-fly in collator)
    print("[INFO] Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", streaming=True, trust_remote_code=True)

    # Create data collator (handles tokenization + masking)
    print("[INFO] Creating diffusion data collator...")
    diffusion_collator = DiffusionCollator(tokenizer, config)

    # Initialize model
    print(f"[INFO] Loading model from {config.MODEL_NAME}...")
    model = RobertaForMaskedLM.from_pretrained(config.MODEL_NAME)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        remove_unused_columns=False,  # Keep dataset columns for custom collator
    )

    # Initialize trainer
    print("[INFO] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        data_collator=diffusion_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    trainer.train()

    # Save model and tokenizer
    print(f"\n[INFO] Saving model to {config.OUTPUT_DIR}...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("[SUCCESS] Training complete!")

    # Run inference test
    run_inference_test(model, tokenizer, dataset, diffusion_collator)


if __name__ == "__main__":
    main()
