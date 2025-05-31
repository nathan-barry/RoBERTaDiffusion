#!/usr/bin/env python3
import sys
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# 1) Parse command-line
if len(sys.argv) < 2:
    print('Usage: python gpt2_generate_debug.py "<YOUR PROMPT HERE>"')
    sys.exit(1)
prompt_text = sys.argv[1]

# 2) Device selection (including MPS for Apple Silicon)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("[INFO] Using MPS (Apple silicon) backend")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[INFO] Using CUDA backend")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU backend")

print("[INFO] Loading GPT-2 tokenizer and model…")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# GPT-2 has no pad_token by default. Set pad_token = eos_token so we get a valid attention_mask.
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully.")

# 4) Tokenize the prompt (explicitly return attention_mask)
print("[INFO] Tokenizing prompt…")
encoding = tokenizer(
    prompt_text,
    return_tensors="pt",
    padding=False,
    truncation=True,
    return_attention_mask=True,
)
input_ids = encoding["input_ids"].to(DEVICE)  # shape: (1, L_prompt)
attention_mask = encoding["attention_mask"].to(DEVICE)  # shape: (1, L_prompt)
print(f"[INFO] Prompt token length = {input_ids.shape[-1]}")

# 5) Generate continuation with sampling
print("[INFO] Starting text generation…")
try:
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
except Exception as e:
    print("[ERROR] model.generate() raised an exception:")
    print(e)
    print("[INFO] Please check PyTorch version / model compatibility.")
    sys.exit(1)

# 6) Decode & print the generated text
print("[INFO] Decoding generated tokens…")
try:
    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
except Exception as e:
    print("[ERROR] tokenizer.decode() crashed:")
    print(e)
    sys.exit(1)

print("\n=== Generated Text ===\n")
print(generated_text)
print("\n======================\n")
