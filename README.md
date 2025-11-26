# RoBERTa Diffusion Text Generation

A research project exploring diffusion-based text generation using RoBERTa as an alternative to traditional autoregressive language models.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync
```

## Usage

### Basic Text Generation

Generate text using the RoBERTa diffusion model:

```bash
uv run python inference.py "Your prompt text here"
```

The script will:
- Use your prompt as the fixed prefix
- Generate continuation using confidence-based parallel decoding
- Display step-by-step progress in terminal
- Create an animated visualization of the generation process

**Skip animation:**
```bash
uv run python inference.py "Your prompt" --animation
```

### GPT-2 Baseline

For comparison, generate text using standard GPT-2:

```bash
uv run python gpt2_inference.py "Your prompt text here"
```

### Side-by-Side Comparison

Run both models simultaneously and compare outputs:

```bash
uv run python compare.py "Your prompt text here"
```

Optionally specify a custom RoBERTa model:
```bash
uv run python compare.py "Your prompt" --roberta-dir path/to/model
```

This creates a synchronized animation showing:
- RoBERTa diffusion generation steps
- GPT-2 autoregressive generation
- Timing metrics for both approaches

### Fine-tuning

Train your own RoBERTa diffusion model:

```bash
uv run python finetune.py
```

**Training Details:**
- **Dataset**: OpenWebText (large-scale web text corpus)
- **Lazy Loading**: Data is tokenized on-the-fly during training
- **Custom Collator**: Handles tokenization and variable masking per batch
- **Prefix Preservation**: First `PREFIX_LEN` tokens are never masked
- **Variable Masking**: Trains on all masking ratios from 0% to 100%

## Project Structure

```
RoBERTaDiffusion/
├── inference.py         # RoBERTa diffusion inference (confidence-based)
├── gpt2_inference.py    # GPT-2 baseline inference
├── compare.py           # Side-by-side model comparison
├── finetune.py          # RoBERTa diffusion training script
├── utils.py             # Shared utilities (device selection, etc.)
├── pyproject.toml       # Project dependencies
└── weights/             # Pre-trained model checkpoints
```
