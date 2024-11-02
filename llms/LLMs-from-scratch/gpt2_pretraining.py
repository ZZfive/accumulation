import math
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from gpt2 import GPTModel, generate_text_simple
from utils import create_dataloader_v1, text2id, id2text

GPT_CONFIG = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}


def calc_loss_bath(input_batch: torch.Tensor, target_batch: torch.Tensor,
                   model: nn.Module, device: torch.device) -> torch.Tensor:
    input_batch, target_batch= input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),
                                             target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: DataLoader, model: nn.Module, device: torch.device, num_batches: int = None) -> float:
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None or num_batches > len(data_loader):
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_bath(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                       device: torch.device, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str,
                       tokenizer: tiktoken.Encoding) -> Tuple[List[float], List[float], List[int]]:
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_bath(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str,
                tokenizer: tiktoken.Encoding, warmp_steps: int, initial_lr: float = 3e-5, min_lr: float = 1e-6) -> Tuple[List[float], List[float], List[int]]:
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * num_epochs
    lr_increment = (peak_lr - initial_lr) / warmp_steps

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmp_steps:  # warmup之前学习率线性增加
                lr = initial_lr + global_step * lr_increment
            else:  # warmup之后学习率cosine降低
                progress = (global_step - warmp_steps) / (total_training_steps - warmp_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_bath(input_batch, target_batch, model, device)
            loss.backward()

            if global_step > warmp_steps:  # warmup之后进行梯度剪裁
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            token_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, eval_iter: int) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: nn.Module, tokenizer: tiktoken.Encoding,
                              device: torch.device, start_context: str) -> None:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text2id(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded_text = id2text(token_ids, tokenizer)
    print(decoded_text)
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int,
             temperature: float = 0.0, top_k : int = None, eos_id: int = None) -> torch.Tensor:

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


if __name__ == "__main__":
    file_path = r"D:\git_github\self\accumulation\llms\LLMs-from-scratch\the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(425)
    train_loader = create_dataloader_v1(train_data,
                                        batch_size=2,
                                        max_length=GPT_CONFIG["context_length"],
                                        stride=GPT_CONFIG["context_length"],
                                        drop_last=True,
                                        shuffle=True,
                                        num_workers=0)
    val_loader = create_dataloader_v1(val_data,
                                      batch_size=2,
                                      max_length=GPT_CONFIG["context_length"],
                                      stride=GPT_CONFIG["context_length"],
                                      drop_last=False,
                                      shuffle=False,
                                      num_workers=0)
    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    model = GPTModel(GPT_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    #     train_loss = calc_loss_loader(train_loader, model, device)
    #     val_loss = calc_loss_loader(val_loader, model, device)

    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    model.to("cpu")
    token_ids = generate(
        model=model,
        idx=text2id("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", id2text(token_ids, tokenizer))