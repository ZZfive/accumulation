"""
DPO训练建立在SFT的基础上，策略模型policy_model和参考模型reference_model初始都是SFT模型，policy_model会更新，reference_model不会更新
"""

import torch
import torch.nn.functional as F

from gpt2 import GPTModel


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
        num_batches = len(num_batches)
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


if __name__ == "__main__":
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