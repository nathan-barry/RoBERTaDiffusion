import os
import torch

# 1) GPT
from gpt import (
    GPTLanguageModel,
    encode as gpt_encode,
    decode as gpt_decode,
    device as gpt_device,
)
from roberta import (
    RoBERTaForMaskedLM,
    get_batch as roberta_get_batch,
    decode as rob_decode,
    itos as rob_itos,
    device as rob_device,
    mask_token_id,
)
from diffusion import MaskedLanguageDiffusion, mask_batch_dynamic, device as diff_device

# All devices should be the same—just pick one
device = torch.device(gpt_device)

weights_dir = "weights"

# ── GPT Inference ─────────────────────────────────────────────────────────────

gpt_model = GPTLanguageModel().to(device)
gpt_ckpt = torch.load(os.path.join(weights_dir, "gpt_weights.pt"), map_location=device)
gpt_model.load_state_dict(gpt_ckpt)
gpt_model.eval()

prompt = "\n"
input_ids = torch.tensor([gpt_encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    out = gpt_model.generate(input_ids, max_new_tokens=100)
generated = gpt_decode(out[0].tolist())

print("=== GPT Generation ===")
print(f"Prompt: {prompt!r}")
print(generated)
print("\n" + "=" * 70 + "\n")

# ── RoBERTa Masked‐LM Inference ────────────────────────────────────────────────

roberta = RoBERTaForMaskedLM().to(device)
rob_ckpt = torch.load(
    os.path.join(weights_dir, "roberta_weights.pt"), map_location=device
)
roberta.load_state_dict(rob_ckpt)
roberta.eval()

print("=== RoBERTa Masked‐LM Samples ===\n")
xb, _ = roberta_get_batch("val")  # xb: (B, T)
logits, _ = roberta(xb, None)  # ignore labels here
preds = torch.argmax(logits, dim=-1)  # (B, T)

inp_ids = xb[0].tolist()
pred_ids = preds[0].tolist()

# build strings, using ❔ for the mask token
masked_str = "".join(
    "�" if tok_id == mask_token_id else rob_itos[tok_id] for tok_id in inp_ids
)
pred_str = "".join(rob_itos[tok_id] for tok_id in pred_ids)

print(f"Input:\n{masked_str}\n\n")
print(f"Output:\n{pred_str}")
print("-" * 60)

# ── Diffusion Denoising Inference ──────────────────────────────────────────────

# # load diffusion checkpoint
# diff_ckpt = torch.load(
#     os.path.join(weights_dir, "diffusion_weights.pt"), map_location=device
# )
# n_steps = len(diff_ckpt["model_states"])
# diffusion = MaskedLanguageDiffusion(n_steps=n_steps).to(device)
# diffusion.mask_probs = diff_ckpt["mask_probs"]
# for model, st in zip(diffusion.models, diff_ckpt["model_states"]):
#     model.load_state_dict(st)
# diffusion.eval()
#
# # take one example from val, fully mask it, then denoise
# xb_val, _ = roberta_get_batch("val")
# x0 = xb_val[:1]  # take first in batch
# xT, _ = mask_batch_dynamic(x0, mask_prob=1.0)  # fully masked
#
# with torch.no_grad():
#     trajectory = diffusion.sample(xT)  # list of [x0_pred,...,xT]
#
# print("=== Diffusion Denoising Trajectory ===")
# for t, x in enumerate(trajectory):
#     seq = x[0].tolist()
#     text = "".join(rob_itos[i] for i in seq)
#     print(f"Step {t:2d} (mask={diffusion.mask_probs[t]:.2f}):")
#     print(text)
#     print("-" * 50)
