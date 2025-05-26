import torch
from pathlib import Path

### Data
# "We consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units,
# without any additional preprocessing or tokenization of the input. This adds approximately
# 15M and 20M additional parameters for BERT-BASE and BERT-LARGE, respectively."
# VOCAB_SIZE = 50_000
VOCAB_SIZE = 30_000
VOCAB_DIR = Path(__file__).parent/"bookcorpus_vocab"
MAX_LEN = 512

### Architecture
N_LAYERS = 6
N_HEADS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384 * 4

### Regularization
DROP_PROB = 0.1

### Masked Language Model
SELECT_PROB = 0.15
MASK_PROB = 0.8
RANDOMIZE_PROB = 0.1

### Optimizer
MAX_LR = 6e-4 # "Peak Learning Rate"
BETA1 = 0.9 # "Adam $\beta_{1}$"
BETA2 = 0.98 # "Adam $\beta_{2}$"
EPS = 1e-6 # "Adam $\epsilon$"
# WEIGHT_DECAY = 0.01
WEIGHT_DECAY = 0 # "Weight Decay"
N_WARM_STEPS = 24_000 # "Warmup Steps"

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
N_WORKERS = 4
CKPT_DIR = Path(__file__).parent/"pretrain/checkpoints"
N_CKPT_SAMPLES = 100_000
