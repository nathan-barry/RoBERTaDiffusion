import os
import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
)

# 1) Hyperparameters
N_STEPS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 256

# linearly spaced mask probabilities from 1/N_STEPS → 1.0
mask_probs = [(i + 1) / N_STEPS for i in range(N_STEPS)]

# 2) Load WikiText-2 and drop empty lines
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
for split in ["train", "validation"]:
    dataset[split] = dataset[split].filter(lambda ex: ex["text"].strip() != "")

# 3) Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def tokenize_function(examples):
    # no padding here; we'll group into fixed-size chunks later
    return tokenizer(
        examples["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding=False,
    )


# 4) Tokenize (no padding)
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# 5) Group into non-overlapping blocks of MAX_LEN
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // MAX_LEN) * MAX_LEN
    result = {
        k: [concatenated[k][i : i + MAX_LEN] for i in range(0, total_length, MAX_LEN)]
        for k in concatenated.keys()
    }
    # Add attention_mask = 1 for all tokens (we never actually used padding)
    result["attention_mask"] = [[1] * MAX_LEN for _ in range(len(result["input_ids"]))]
    return result


tokenized = tokenized.map(
    group_texts, batched=True, remove_columns=tokenized["train"].column_names
)

# 6) Loop over each mask probability
for idx, prob in enumerate(mask_probs):
    run_name = f"roberta-diffusion-{idx:02d}-{prob:.2f}"
    print(f"\n=== Finetuning mask_prob={prob:.2f} → output_dir={run_name} ===\n")
    os.makedirs(run_name, exist_ok=True)

    # 6a) Fresh model for this noise level
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    # 6b) Custom collator: masks exactly `prob` fraction of tokens (no 80/10/10)
    def custom_collator(features):
        """
        features: list of dicts with 'input_ids' and 'attention_mask' (both lists of length MAX_LEN)
        Returns a batch dict with:
          - input_ids: LongTensor (B, MAX_LEN) masked on CPU
          - attention_mask: LongTensor (B, MAX_LEN) unchanged on CPU
          - labels: LongTensor (B, MAX_LEN) where non-masked = -100, masked = original token IDs, on CPU
        """
        # Build CPU tensors
        batch_input_ids = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        batch_attention = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )

        labels = batch_input_ids.clone()

        # Create a random mask array on CPU
        rand = torch.rand(batch_input_ids.shape)

        # Build mask_candidate: True where attention_mask=1 AND token not special
        special_ids = set(tokenizer.all_special_ids)
        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in special_ids:
            is_special |= batch_input_ids == sid
        mask_candidate = (batch_attention == 1) & (~is_special)

        # Now sample which positions to mask
        mask_positions = (rand < prob) & mask_candidate

        # Replace masked positions with <mask> token
        batch_input_ids[mask_positions] = tokenizer.mask_token_id

        # Set labels: only compute loss on masked positions
        labels[~mask_positions] = -100

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
        }

    # 6c) Training arguments
    training_args = TrainingArguments(
        output_dir=run_name,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="epoch",  # only save at epoch end
        save_total_limit=1,
        logging_steps=200,
    )

    # 6d) Create Trainer with custom collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=custom_collator,
        tokenizer=tokenizer,
    )

    # 6e) Train & save
    trainer.train()
    trainer.save_model(run_name)
    tokenizer.save_pretrained(run_name)
    print(f"✅ Finished run {run_name}\n")

    # 7) Quick inference sample for this mask_prob
    model.eval()
    block = tokenized["validation"][0]
    features = {
        "input_ids": block["input_ids"],
        "attention_mask": block["attention_mask"],
    }
    batch = custom_collator([features])
    input_ids_masked = batch["input_ids"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids_masked).logits
        pred_ids = logits.argmax(dim=-1)  # <— corrected here

    # decode input (replace <mask> with ❔) and prediction
    masked_str = tokenizer.decode(
        input_ids_masked[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace(tokenizer.mask_token, "❔")
    pred_str = tokenizer.decode(
        pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"--- Sample at mask_prob={prob:.2f} ---")
    print("Input :\n" + masked_str + "\n")
    print("Output:\n" + pred_str)
    print("-" * 60)

print("All diffusion‐style finetuning runs complete.")
