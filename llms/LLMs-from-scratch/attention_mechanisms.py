
import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.w_q = nn.Parameter(torch.rand(d_in, d_out))
        self.w_k = nn.Parameter(torch.rand(d_in, d_out))
        self.w_v = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # 计算查询、键和值
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # 计算注意力分数
        attn_scores = q @ k.T

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)

        # 计算输出
        context_vec = attn_weights @ v

        return context_vec
    

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, qkv_bias: bool = False) -> None:
        """
        初始化因果注意力机制模块。

        参数:
        - d_in: 输入维度
        - d_out: 输出维度
        - context_length: 上下文长度
        - dropout: Dropout比率
        - qkv_bias: 是否为查询、键、值线性层添加偏置
        """
        super().__init__()
        self.d_out = d_out
        # 创建查询、键、值的线性变换层
        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 创建dropout层
        self.dropout = nn.Dropout(dropout)
        # 创建上三角掩码，用于实现因果注意力
        self.register_buffer("mask",
                             torch.triu(torch.zeros(context_length, context_length),
                                        diagonal=1))
        
    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入张量，形状为 (batch_size, num_tokens, d_in)

        返回:
        - context_vec: 注意力计算后的上下文向量
        """
        b, num_tokens, d_in = x.shape
        # 计算键、查询、值
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(1, 2)
        # 应用因果掩码
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 应用dropout
        attn_weights = self.dropout(attn_weights)
        # 计算上下文向量
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        """
        多头注意力包装器的初始化函数。

        参数:
        - d_in: 输入维度
        - d_out: 每个头的输出维度
        - context_length: 上下文长度
        - dropout: Dropout比率
        - num_heads: 注意力头的数量
        - qkv_bias: 是否为查询、键、值线性层添加偏置
        """
        super().__init__()
        # 创建多个因果注意力头
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入张量，形状为 (batch_size, num_tokens, d_in)

        返回:
        - 多头注意力的输出，形状为 (batch_size, num_tokens, d_out * num_heads)
        """
        # 对每个注意力头应用输入，然后在最后一个维度上拼接结果
        return torch.cat([head(x) for head in self.heads], dim=-1)  # 最终输出的token维度是原始token维度的num_heads倍
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkvb_bias=False):
        super.__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # 创建查询、键、值的线性变换层
        self.w_q = nn.Linear(d_in, d_out, bias=qkvb_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkvb_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkvb_bias)
        # 创建输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        # 创建dropout层
        self.dropout = nn.Dropout(dropout)
        # 创建上三角掩码
        self.register_buffer("mask",
                             torch.triu(torch.zeros(context_length, context_length),
                                        diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 计算查询、键、值
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 重塑张量以适应多头注意力
        q = quit.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整张量维度顺序
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn_scores = q @ k.transpose(2, 3)
        # 应用因果掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 计算上下文向量
        context_vec = (attn_weights @ v).transpose(1, 2)
        # 重塑并连接多头输出
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 应用输出投影
        context_vec = self.out_proj(context_vec)
        return context_vec