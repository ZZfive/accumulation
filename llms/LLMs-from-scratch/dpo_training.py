"""
DPO训练建立在SFT的基础上，策略模型policy_model和参考模型reference_model初始都是SFT模型，policy_model会更新，reference_model不会更新
"""

import torch
import tiktoken
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpt2 import GPTModel
from gpt2_pretraining import generate_and_print_sample
from utils import format_input, PreferenceDataset, custom_collate_fn_dpo


def compute_dpo_loss(
        model_chosen_logprobs,
        model_rejected_logprobs,
        reference_chosen_logprobs,
        reference_rejected_logprobs,
        beta: float = 0.1
    ):
    """计算一批策略模型和参考模型对数概率的DPO损失。

    Args:
        model_chosen_logprobs: 策略模型为选中响应的对数概率。形状: (batch_size,)
        model_rejected_logprobs: 策略模型为被拒绝响应的对数概率。形状: (batch_size,)
        reference_chosen_logprobs: 参考模型为选中响应的对数概率。形状: (batch_size,)
        reference_rejected_logprobs: 参考模型为被拒绝响应的对数概率。形状: (batch_size,)
        beta: DPO损失的温度参数；通常在0.1到0.5的范围内。随着beta接近0，我们忽略参考模型。
        label_smoothing: DPO损失的保守性。

    Returns:
        一个包含三个张量的元组: (loss, chosen_rewards, rejected_rewards)。
    """
    model_logprobs = model_chosen_logprobs - model_rejected_logprobs
    reference_logprobs = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logprobs - reference_logprobs

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    reference_rewards = (reference_chosen_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), reference_rewards.mean()


def compute_logprobs(logits, labels, selection_mask: bool = None):
    """
    计算对数概率。

    Args:
      logits: 形状为 (batch_size, num_tokens, vocab_size) 的张量
      labels: 形状为 (batch_size, num_tokens) 的张量
      selection_mask: 形状为 (batch_size, num_tokens) 的张量

    Returns:
      mean_log_prob: 排除填充标记后的平均对数概率。
    """
    # labels是从输入向后移动一位得到
    labels = labels[:, 1:].clone()

    # 因为labels要想后移动以为，那么输入就不取最后一位
    logits = logits[:, :-1, :]

    # 沿着最后一个维度计算logits的softmax，并取对数
    log_probs = F.log_softmax(logits, dim=-1)
     
    # 根据labels从logits中选择对应的对数概率
    selected_log_probs = torch.gather(  # gather本质是一个选择函数返回一个新张量，其中每个位置上值是基于index从input中选择的
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # 如果提供了选择掩码
    if selection_mask is not None:
        # 复制并移除选择掩码的第一个元素
        mask = selection_mask[:, 1:].clone()

        # 使用掩码过滤selected_log_probs
        selected_log_probs = selected_log_probs * mask

        # 计算掩码区域内的平均对数概率
        avg_log_probs = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_probs
    else:
        # 如果没有提供选择掩码，则直接计算selected_log_probs的平均值
        return selected_log_probs.mean(-1)
    

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta: float = 0.1):
    policy_chosen_log_probs = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )

    policy_rejected_log_probs = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )

    ref_chosen_log_probs = compute_logprobs(
        logits=reference_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )

    ref_rejected_log_probs = compute_logprobs(
        logits=reference_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probs,
        model_rejected_logprobs=policy_rejected_log_probs,
        reference_chosen_logprobs=ref_chosen_log_probs,
        reference_rejected_logprobs=ref_rejected_log_probs,
        beta=beta
    )

    return loss, chosen_rewards, rejected_rewards


def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta: float = 0.1, num_batches: int = None):
    total_loss, total_chosen_rewards, total_rejected_rewards = 0, 0, 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards


def eval_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, eval_iter, beta: float = 0.1):
    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_rewards": train_chosen_rewards,
        "train_rejected_rewards": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_rewards": val_chosen_rewards,
        "val_rejected_rewards": val_rejected_rewards,
    }

    policy_model.train()
    return res


def train_model_dpo_simple(policy_model, reference_model, train_loader, val_loader, optimizer,
                           num_epochs, beta, eval_freq, eval_iter, start_context, tokenizer):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }

    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train()

        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, _, _ = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()
            optimizer.step()

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            if global_step % eval_freq == 0:
                res = eval_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )

                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model = GPTModel(BASE_CONFIG)

    model.load_state_dict(
        torch.load(
            "gpt2-medium355M-sft.pth",  # SFT过程训练保存的模型权重文件
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    model.eval()  # 初始时模型是eval模式

    policy_model = model

    reference_model = GPTModel(BASE_CONFIG)
    reference_model.load_state_dict(
        torch.load(
            "gpt2-medium355M-sft.pth",
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    reference_model.eval()

    policy_model.to(device)
    reference_model.to(device)

    num_workers = 0
    batch_size = 8

    train_data = None  # 自行构建
    val_data = None

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn_dpo,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn_dpo,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # 只将policy_model的参数传给optimizer，使用非常小的学习率
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)
    num_epochs = 1  # 只训练一个epoch，因为dpo训练很容易崩溃，即使损失降低，但模型生成变差
    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=None,
        val_loader=None,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.1, # value between 0.1 and 0.5；beta值可以从0.1增加到0.5来减少DPO的影响（我们这里使用0.1是为了让结果更明显）
        eval_freq=5,
        eval_iter=5,
        start_context=format_input("一段测试文本"),
        tokenizer=tokenizer
    )