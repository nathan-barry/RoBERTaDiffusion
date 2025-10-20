# RoBERTa Diffusion Text Generation

A research project exploring diffusion-based text generation using RoBERTa as an alternative to traditional autoregressive language models.

## Overview

Instead of generating text left-to-right one token at a time (like GPT-2), this project uses a **diffusion/denoising approach**:

1. Start with a fixed text prefix (first 16 tokens)
2. Fill remaining positions with mask tokens
3. Iteratively predict and reveal tokens over multiple steps
4. Progressively decrease masking probability until fully denoised

This approach leverages RoBERTa's bidirectional context understanding to generate text through iterative refinement rather than sequential prediction.

## Key Features

- **Bidirectional Generation**: Unlike autoregressive models, can attend to full sequence context
- **Iterative Denoising**: Gradually reveals text over configurable number of steps
- **Prefix Control**: First 16 tokens remain fixed, providing stable context
- **Visual Animations**: Step-by-step matplotlib animations showing generation process
- **Comparative Analysis**: Side-by-side comparison with GPT-2 baseline

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync
```

**Requirements:**
- Python >= 3.11
- PyTorch 2.7.0+
- Transformers 4.52.4
- Datasets 3.6.0
- Matplotlib 3.10.3
- Accelerate 1.7.0

## Usage

### Basic Text Generation

Generate text using the RoBERTa diffusion model:

```bash
python inference.py
```

The script will:
- Prompt you for input text
- Generate continuation using iterative denoising
- Display step-by-step progress in terminal
- Create an animated visualization of the generation process

**Parameters** (edit in `inference.py`):
- `MAX_LEN`: Maximum sequence length (default: 256)
- `PREFIX_LEN`: Number of fixed prefix tokens (default: 16)
- `N_STEPS`: Number of denoising iterations (default: 10)
- `TOP_K`: Top-k sampling parameter (default: 50)
- `TOP_P`: Nucleus sampling parameter (default: 0.95)

### GPT-2 Baseline

For comparison, generate text using standard GPT-2:

```bash
python gpt2_inference.py
```

### Side-by-Side Comparison

Run both models simultaneously and compare outputs:

```bash
python compare.py
```

This creates a synchronized animation showing:
- RoBERTa diffusion generation steps
- GPT-2 autoregressive generation
- Timing metrics for both approaches

### Fine-tuning

Train your own RoBERTa diffusion model:

```bash
python finetune.py
```

**Training Details:**
- Dataset: WikiText-2 (configurable)
- Epochs: 30 (default)
- Batch size: 16
- Custom diffusion collator with variable masking
- Preserves first 16 tokens from masking

## Project Structure

```
RoBERTaDiffusion/
    inference.py            # Main RoBERTa diffusion inference
    finetune.py             # Training script
    compare.py              # RoBERTa vs GPT-2 comparison
    gpt2_inference.py       # GPT-2 baseline
    pyproject.toml          # Dependencies
    weights/                # Pre-trained models
```
