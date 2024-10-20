import os
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from llama2 import compute_rope, FeedForward, RMSNorm
from utils import model_memory_size

'''
基于llama2调正得到llama3主要有以下几步
1、调整ROPE
2、定义了一个用于存储mask、cos、sin的buffer类
3、实现GQA
4、将以上更新调整至TransformerBlock，重新定义llama3
'''


def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None):
    """
    预计算ROPE参数。

    Args:
    - head_dim (int): embedding维度，必须是偶数。
    - theta_base (int, optional): ROPE参数的基础值，默认为10000。
    - context_length (int, optional): 上下文长度，默认为4096。
    - freq_config (dict, optional): 频率配置，包含调整ROPE参数的配置。

    Returns:
    - cos (torch.Tensor): 余弦值。
    - sin (torch.Tensor): 正弦值。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    if freq_config is not None:
        # 根据频率配置调整逆频率
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factior"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factior"]

        # 计算波长
        wavelen = 2 * torch.pi / inv_freq

        # 根据波长调整逆频率
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 计算平滑因子
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_context_length"] - freq_config["low_freq_factor"]
        )

        # 计算平滑后的逆频率
        smooth_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 判断是否为中频率
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smooth_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # 生成位置序列
    positions = torch.arange(context_length)
    
    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]

    # 将角度扩展到两个维度
    angles = torch.cat([angles, angles], dim=1)

    # 计算余弦和正弦值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


class ShareBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in ShareBuffers._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            ShareBuffers._buffers[key] = (mask, cos, sin)

        return ShareBuffers._buffers[key]
    

class GroupQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads,
                 num_kv_groups, rope_base=10000, rope_config=None, dtype=None):
        """
        初始化GroupQueryAttention模块。

        Args:
        - d_in (int): 输入维度。
        - d_out (int): 输出维度。
        - context_length (int): 上下文长度。
        - num_heads (int): 注意力头的数量。
        - num_kv_groups (int): 键值对的分组数量。
        - rope_base (int, optional): ROPE参数的基础值，默认为10000。
        - rope_config (dict, optional): ROPE配置。
        - dtype (torch.dtype, optional): 数据类型，默认为None。
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads munst be divisible by num_kv_groups"

        self.d_out= d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 定义线性层，用于计算查询、键和值
        self.w_k = nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # 定义线性层，用于计算查询和输出
        self.w_q = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_in, d_out, bias=False, dtype=dtype)

        # 获取预计算的ROPE参数
        mask, cos, sin = ShareBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)

        # 将预计算的ROPE参数注册为模型的buffer
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        """
        前向传播函数。

        Args:
        - x (torch.Tensor): 输入张量。

        Returns:
        - torch.Tensor: 输出张量。
        """
        # 获取输入张量的形状
        b, num_tokens, d_in = x.shape
        # 计算查询、键和值
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 调整查询、键和值的形状
        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_kv_groups, self.head_dim)  # k，v的维度与q不同，第2维度是比q小的，因为num_kv_groups = num_heads / group_size
        v = v.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # 转置键和值
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        # 计算ROPE
        k = compute_rope(k, self.cos, self.sin)
        q = compute_rope(q, self.cos, self.sin)

        # 重复键和值
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 计算注意力分数
        attn_scores = q @ k.transpose(2, 3)

        # 创建掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 对注意力分数进行掩码填充
        attn_scores.mask_fill_(mask_bool, -torch.inf)
        assert k.shape[-1] == self.head_dim

        # 计算上下文向量
        context_vec = (attn_scores @ v).transpose(1, 2)

        # 调整上下文向量的形状
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 通过输出投影层得到最终的上下文向量
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut

        return x


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        # batch_size, seq_len = in_idx.shape
        token_emb = self.token_emb(in_idx)
        x = token_emb
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
    

class Tokenizer:
    """
    Tokenizer类用于对文本进行编码和解码。
    """
    def __init__(self, model_path):
        """
        初始化Tokenizer类。

        Args:
        - model_path (str): 模型文件路径。
        """
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )


    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        """
        对文本进行编码。

        Args:
        - text (str): 要编码的文本。
        - bos (bool, optional): 是否在编码的开头添加特殊token "<|begin_of_text|>"。默认为False。
        - eos (bool, optional): 是否在编码的结尾添加特殊token "<|end_of_text|>"。默认为False。
        - allowed_special (set, optional): 允许的特殊token集合。默认为空集。
        - disallowed_special (tuple, optional): 不允许的特殊token元组。默认为空元组。

        Returns:
        - tokens (list): 编码后的token列表。
        """
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        """
        对token列表进行解码。

        Args:
        - tokens (list): 要解码的token列表。

        Returns:
        - text (str): 解码后的文本。
        """
        return self.model.decode(tokens)


if __name__ == "__main__":
    LLAMA3_CONFIG_8B = {
        "vocab_size": 128256,    # NEW: Larger vocabulary size
        "context_length": 8192,  # NEW: Larger context length
        "emb_dim": 4096,         # Embedding dimension
        "n_heads": 32,           # Number of attention heads
        "n_layers": 32,          # Number of layers
        "hidden_dim": 14336,     # NEW: Larger size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,        # NEW: Key-Value groups for grouped-query attention
        "rope_base": 50000,      # NEW: The base in RoPE's "theta" was increased to 50_000
        "rope_freq": None,       # NEW: Additional configuration for adjusting the RoPE frequencies
        "dtype": torch.bfloat16  # Lower-precision dtype to save memory
    }

    model = Llama3Model(LLAMA3_CONFIG_8B)

    print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)
    print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)
    print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")


    # LLama3.1与LLama3在架构上没有区别，主要是RoPE的频率有一些调整，可以通过调整config实现
    LLAMA31_CONFIG_8B = {
        "vocab_size": 128256,       # Vocabulary size
        "context_length": 131072,   # NEW: Larger supported context length
        "emb_dim": 4096,            # Embedding dimension
        "n_heads": 32,              # Number of attention heads
        "n_layers": 32,             # Number of layers
        "hidden_dim": 14336,        # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
        "rope_base": 50000,         # The base in RoPE's "theta"
        "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
        "rope_freq": {              # NEW: RoPE frequency scaling
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }

    # LLama3.2与LLama3.1在架构上基本没有区别，可以通过调整config实现
    LLAMA32_CONFIG_1B = {
        "vocab_size": 128256,       # Vocabulary size
        "context_length": 131072,   # Context length
        "emb_dim": 2048,            # NEW: Half the embedding dimension
        "n_heads": 32,              # Number of attention heads
        "n_layers": 16,             # NEW: Half the number of layers
        "hidden_dim": 8192,         # NEW: Almost half the size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
        "rope_base": 50000,         # The base in RoPE's "theta"
        "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
        "rope_freq": {              # RoPE frequency scaling
            "factor": 32.0,         # NEW: Adjustment of the rescaling factor
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }