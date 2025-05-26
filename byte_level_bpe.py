import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizerFast

import config
from utils import REGEX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epubtxt_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def parse(epubtxt_dir, with_document=False):
    print("Parsing BookCorpus...")
    lines = list()
    for doc_path in tqdm(list(Path(epubtxt_dir).glob("*.txt"))):
        with open(doc_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or (line.count(" ") < 1):
                    continue
                if not with_document:
                    lines.append(line)
                else:
                    lines.append((doc_path.name, line))
    print("Completed.")
    print(f"Number of paragraphs: {len(lines):,}")
    return lines


def train_fast_roberta_tokenizer(corpus, vocab_size, vocab_dir):
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=vocab_size, length=len(corpus))
    tokenizer.save_pretrained(vocab_dir)


def load_fast_roberta_tokenizer(vocab_dir):
    tokenizer = RobertaTokenizerFast.from_pretrained(vocab_dir)
    return tokenizer


if __name__ == "__main__":
    args = get_args()

    lines = parse(args.epubtxt_dir)
    tokenizer = train_fast_roberta_tokenizer(
        corpus=lines,
        vocab_size=config.VOCAB_SIZE,
        vocab_dir=config.VOCAB_DIR,
    )
