import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import os
from datasets import load_dataset
from transformers import RobertaTokenizer

from model import RoBERTaForPretraining


class CustomDataset(Dataset):
    def __init__(self, tokenizer, max_len, split="train"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encodings = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze(0)  # shape (max_len,)
        attention_mask = encodings['attention_mask'].squeeze(0)  # shape (max_len,)
        return input_ids, attention_mask


def compute_loss(model, input_ids, attention_mask):
    # The forward pass in the RoBERTa model already handles MLM loss internally
    logits = model(input_ids)
    # Compute CrossEntropy loss only for the non-masked tokens
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), ignore_index=0)
    return loss


def train(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc="Training", ncols=100, leave=True)
    
    for batch in progress_bar:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        optimizer.zero_grad()

        loss = compute_loss(model, input_ids, attention_mask)
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (len(progress_bar)))
        
    return running_loss / len(train_dataloader)


def main():
    # Configuration
    max_len = 128
    pad_id = 0
    batch_size = 32
    num_epochs = 3
    learning_rate = 2e-5
    n_layers = 12
    n_heads = 12
    hidden_size = 768
    mlp_size = 768 * 4
    drop_prob = 0.1

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    vocab_size = tokenizer.vocab_size  # This will be 50265 for roberta-base
    
    # Initialize model
    model = RoBERTaForPretraining(
        vocab_size=vocab_size,
        max_len=max_len,
        pad_id=pad_id,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare dataset and dataloader
    train_dataset = CustomDataset(tokenizer=tokenizer, max_len=max_len, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint_dir = "checkpoints"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"roberta_epoch_{epoch+1}.pth"))
    

if __name__ == "__main__":
    main()
