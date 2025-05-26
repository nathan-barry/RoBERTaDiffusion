import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from pathlib import Path
from time import time
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt

import config
from utils import get_elapsed_time
from model import RoBERTaForPretraining
from byte_level_bpe import load_fast_roberta_tokenizer
from bookcorpus import BookCorpusForRoBERTa
from masked_language_model import MaskedLanguageModel
from loss import PretrainingLoss
from evalute import get_mlm_acc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epubtxt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=8192) # "Batch Size"
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def get_lr(max_lr, warmup_steps, n_steps, step): # "Learning Rate Decay: Linear"
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        lr = - max_lr / (n_steps - warmup_steps + 1) * (step - n_steps - 1)
    return lr


def vis_lr(max_lr, warmup_steps, n_steps):
    lrs = [
        get_lr(max_lr=max_lr, warmup_steps=warmup_steps, n_steps=n_steps, step=step)
        for step in range(1, n_steps + 1)
    ]
    plt.plot(lrs)
    plt.show()


def update_lr(lr, optim):
    optim.param_groups[0]["lr"] = lr


def save_checkpoint(step, model, optim, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "optimizer": optim.state_dict(),
    }
    if config.N_GPUS > 1:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()
    torch.save(ckpt, str(ckpt_path))


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"N_WORKERS = {config.N_WORKERS}")
    print(f"MAX_LEN = {config.MAX_LEN}")

    N_STEPS = (8192 * 100_000) // args.batch_size
    N_ACCUM_STEPS = 8192 // args.batch_size
    print(f"N_STEPS = {N_STEPS:,}", end="\n\n")

    tokenizer = load_fast_roberta_tokenizer(vocab_dir=config.VOCAB_DIR)
    train_ds = BookCorpusForRoBERTa(
        epubtxt_dir=args.epubtxt_dir,
        tokenizer=tokenizer,
        seq_len=config.MAX_LEN,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = RoBERTaForPretraining( # Smaller than BERT-Base
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        pad_id=train_ds.pad_id,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
    ).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)

    mlm = MaskedLanguageModel(
        vocab_size=config.VOCAB_SIZE,
        mask_id=tokenizer.mask_token_id,
        no_mask_token_ids=[
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        ],
        select_prob=config.SELECT_PROB,
        mask_prob=config.MASK_PROB,
        randomize_prob=config.RANDOMIZE_PROB,
    )

    optim = Adam(
        model.parameters(),
        lr=config.MAX_LR,
        betas=(config.BETA1, config.BETA2),
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY,
    )

    crit = PretrainingLoss(vocab_size=config.VOCAB_SIZE)

    ### Resume
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=config.DEVICE)
        if config.N_GPUS > 1:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        prev_ckpt_path = Path(args.ckpt_path)
        print(f"Resuming from checkpoint '{str(Path(args.ckpt_path).name)}'...")
    else:
        step = 0
        prev_ckpt_path = Path(".pth")

    print("Training...")
    start_time = time()
    accum_loss = 0
    accum_acc = 0
    step_cnt = 0
    while True:
        for gt_token_ids in train_dl:
            if step < N_STEPS:
                step += 1

                lr = get_lr(
                    max_lr=config.MAX_LR,
                    warmup_steps=config.N_WARM_STEPS,
                    n_steps=N_STEPS,
                    step=step,
                )
                update_lr(lr=lr, optim=optim)

                gt_token_ids = gt_token_ids.to(config.DEVICE)
                masked_token_ids, mlm_mask = mlm(gt_token_ids)

                pred_token_ids = model(masked_token_ids)
                loss = crit(
                    pred_token_ids=pred_token_ids,
                    gt_token_ids=gt_token_ids,
                    mlm_mask=mlm_mask,
                )
                accum_loss += loss.item()
                loss /= N_ACCUM_STEPS
                loss.backward()

                if step % N_ACCUM_STEPS == 0:
                    optim.step()
                    optim.zero_grad()

                acc = get_mlm_acc(pred_token_ids=pred_token_ids, gt_token_ids=gt_token_ids)
                accum_acc += acc
                step_cnt += 1

                if (step % (config.N_CKPT_SAMPLES // args.batch_size) == 0) or (step == N_STEPS):
                    print(f"[ {step:,}/{N_STEPS:,} ][ {get_elapsed_time(start_time)} ]", end="")
                    print(f"[ MLM loss: {accum_loss / step_cnt:.4f} ]", end="")
                    print(f"[ MLM acc: {accum_acc / step_cnt:.3f} ]")

                    start_time = time()
                    accum_loss = 0
                    accum_acc = 0
                    step_cnt = 0

                    cur_ckpt_path = config.CKPT_DIR/f"bookcorpus_step_{step}.pth"
                    save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
                    if prev_ckpt_path.exists():
                        prev_ckpt_path.unlink()
                    prev_ckpt_path = cur_ckpt_path
