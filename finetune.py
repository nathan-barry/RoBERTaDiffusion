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
N_STEPS = 10
NUM_EPOCHS = 30
BATCH_SIZE = 16
MAX_LEN = 256
PREFIX_LEN = 16

# linearly spaced mask probabilities from 1/N_STEPS → 1.0
mask_probs = [(i + 1) / N_STEPS for i in range(N_STEPS - 1, -1, -1)]

# 2) Load WikiText-2 and drop empty lines
dataset = load_dataset("openwebtext")
for split in ["train", "validation"]:
    dataset[split] = dataset[split].filter(lambda ex: ex["text"].strip() != "")

# 3) Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer.model_max_length = MAX_LEN  # just to be safe


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding=False,
    )


# 4) Tokenize (no padding)
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)


# 5) Group into non-overlapping blocks of MAX_LEN
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // MAX_LEN) * MAX_LEN

    result = {
        k: [concatenated[k][i : i + MAX_LEN] for i in range(0, total_length, MAX_LEN)]
        for k in concatenated.keys()
    }
    # Since no padding, attention_mask is all 1's
    result["attention_mask"] = [[1] * MAX_LEN for _ in range(len(result["input_ids"]))]
    return result


tokenized = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
)

# 6) Instantiate a single RoBERTaForMaskedLM
model = RobertaForMaskedLM.from_pretrained("roberta-base")


# 7) Custom collator that:
#    - samples one mask-probability p from mask_probs per BATCH
#    - never masks tokens at positions [0 .. PREFIX_LEN-1]
#    - masks proportion p of the *remaining* (non-special, non-prefix) tokens
def diffusion_collator(features):
    """features: list of dicts with 'input_ids' and 'attention_mask'.

    Returns a batch dict:
      - input_ids: (B, MAX_LEN) with some tokens replaced by <mask>
      - attention_mask: (B, MAX_LEN) unchanged
      - labels: (B, MAX_LEN) where unmasked = -100, masked = original token IDs
    """
    # Stack into CPU tensors
    batch_input_ids = torch.tensor(
        [f["input_ids"] for f in features], dtype=torch.long
    )  # shape (B, MAX_LEN)
    batch_attention = torch.tensor(
        [f["attention_mask"] for f in features], dtype=torch.long
    )  # shape (B, MAX_LEN)

    # Clone to be labels; we'll set unmasked → -100 later
    labels = batch_input_ids.clone()  # shape (B, MAX_LEN)

    # 7a) Sample mask probability p for this batch
    p = float(mask_probs[torch.randint(low=0, high=len(mask_probs), size=(1,))])
    # (Option: sample uniformly continuous p = torch.rand(1).item())

    B, L = batch_input_ids.shape  # L should equal MAX_LEN

    # 7b) Build a boolean mask “cannot_mask” for every position that must NOT be masked:
    #      - any special token (CLS, SEP, PAD, etc.)
    #      - any position < PREFIX_LEN
    special_ids = set(tokenizer.all_special_ids)
    is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= batch_input_ids == sid

    # Build a broadcasted row-vector [0, 1, 2, ..., L-1] < PREFIX_LEN
    device = batch_input_ids.device
    pos_idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # shape (B, L)
    is_prefix = pos_idxs < PREFIX_LEN  # True for positions 0..PREFIX_LEN-1

    # Combine to get mask_candidate = everything that *can* be masked:
    #   mask_candidate = (attention_mask == 1) AND (not special) AND (not prefix)
    mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)
    # shape (B, L) of booleans

    # 7c) Draw random numbers uniformly in [0,1) for each token
    rand = torch.rand_like(batch_input_ids, dtype=torch.float)  # shape (B, L)

    # Decide which positions to mask:
    mask_positions = (rand < p) & mask_candidate  # shape (B, L)

    # 7d) Replace those positions with <mask> token in the input
    batch_input_ids[mask_positions] = tokenizer.mask_token_id

    # 7e) For labels, only compute loss where mask_positions is True:
    labels[~mask_positions] = -100  # unmasked positions get -100

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": labels,
    }


# 8) Training arguments
training_args = TrainingArguments(
    output_dir="roberta-diffusion-single-with-prefix",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=200,
    # You can also add gradient_accumulation_steps or increase max_steps if desired
)

# 9) Create a single Trainer with our modified collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=diffusion_collator,
    tokenizer=tokenizer,
)

# 10) Train & save
trainer.train()
trainer.save_model("roberta-diffusion-single-with-prefix")
tokenizer.save_pretrained("roberta-diffusion-single-with-prefix")

print("Finished diffusion‐style finetuning with first 64 tokens never masked\n")

# 11) Quick inference check
model.eval()
block = tokenized["validation"][0]
features = {
    "input_ids": block["input_ids"],
    "attention_mask": block["attention_mask"],
}
batch = diffusion_collator([features])
input_ids_masked = batch["input_ids"].to(model.device)

with torch.no_grad():
    logits = model(input_ids_masked).logits
    pred_ids = logits.argmax(dim=-1)

masked_str = tokenizer.decode(
    input_ids_masked[0],
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
).replace(tokenizer.mask_token, "�")

pred_str = tokenizer.decode(
    pred_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("--- Sample inference (random mask‐rate, prefix unmasked) ---")
print("Input :\n" + masked_str + "\n")
print("Output:\n" + pred_str)
print("-" * 60)
