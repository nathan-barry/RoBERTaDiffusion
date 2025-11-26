# RoBERTa Diffusion Text Generation

A research project exploring diffusion-based text generation using RoBERTa as an alternative to traditional autoregressive language models.

View the related blog post [here](https://nathan.rs/posts/roberta-diffusion/).

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync
```

## Usage

### Fine-tuning

Train your own RoBERTa diffusion model on openwebtext:

```bash
uv run python finetune.py
```

The blog post used `wikitext-2` instead of `openwebtext`, which seems to for some reason lead to better generations. You can mess around and try changing datasets. I originally trained this by renting a H200 for an hour.

### Basic Text Generation

Generate text using the RoBERTa diffusion model:

```bash
uv run python inference.py "Your prompt text here"
```

Depending on what the `PREFIX_LEN` is set to, your prompt will need to be that length to not be out of distribution. It wouldn't be too hard to add variable length prefixes during training though.

To show the generation step-by-step as an animation, just add this flag:

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

**Training Details:**
- **Dataset**: OpenWebText (large-scale web text corpus)
- **Lazy Loading**: Data is tokenized on-the-fly during training (can be changed)
- **Custom Collator**: Handles tokenization and variable masking per batch
- **Prefix Preservation**: First `PREFIX_LEN` tokens are never masked
- **Variable Masking**: Trains on all masking ratios from 0% to 100%

## Project Structure

```
RoBERTaDiffusion/
├── config.py            # configuration for training and inference
├── finetune.py          # RoBERTa diffusion training script
├── inference.py         # RoBERTa diffusion inference
├── gpt2_inference.py    # GPT-2 baseline inference
├── compare.py           # Side-by-side model comparison
```
