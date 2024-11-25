#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vqvae_pytorch.py
@Time    :   2024/11/25 23:47:23
@Author  :   zzfive 
@Desc    :   手敲VQ-VAE，参考链接：https://nbviewer.jupyter.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
'''


'''
概念说明：
1、码本/codebook本质是一个向量矩阵，可以理解为一个embeddings，维度为[K, D], K就是codebook中的向量数量，D是每个向量的维度
2、encoder输出的向量维度与codebook中的向量维度一致，进而可以与codebook的所有向量计算相似度，然后基于最相似的向量的索引构建one-hot向量，
再与codebook相乘，其实就是用codebook中最相似的向量表示，但整个过程类似于LLM中token转为向量的过程
3、encoder输入是与索引相对应的嵌入向量，该向量通过解码器产生重建图像
4、最近邻查找在向后传递中没有真正的梯度，是直接将梯度从解码器传递到编码器。直觉是，由于编码器的输出表示和解码器的输入共享相同的 D 通道维度空间
因此梯度包含关于编码器必须如何改变其输出以降低重建损失的有用信息。
5、训练巡视会包含以下三部分
    重建损失：优化解码器和编码器
    码本损失：codebook训练时的损失，用于该过程没有地图，只用字典学习算法即使用codebook向量与encoder输出的L2范数作为损失
    commitment损失：防止codebook中的向量无限制增长，使用此附加损失进行限制
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 等价于codebook类，其中self._embedding的权重就是最终使用的codebook向量
class VectorQuantizer(nn.Module):
    """
    定义一个向量量化器（Vector Quantizer），用于将输入向量量化到离散的代码书（codebook）中。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float) -> None:
        """
        初始化向量量化器。

        Args:
            num_embeddings (int): 代码书中向量的数量，即K。
            embedding_dim (int): 每个向量的维度，即D。
            commitment_cost (float): 损失函数中的commitment loss的系数，用于防止代码书中的向量无限制增长。
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        # 初始化代码书，使用nn.Embedding模块来存储代码书中的向量。
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 初始化代码书向量的权重，使用均匀分布初始化。
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost  # 损失函数中的commitment loss的系数

    def forward(self, inputs: torch.Tensor):
        """
        向量量化器的前向传播。

        Args:
            inputs (torch.Tensor): 输入的向量，形状为[B, C, H, W]，其中B是批次大小，C是通道数，H和W是高度和宽度。

        Returns:
            loss (torch.Tensor): 总的损失，包括重建损失和commitment损失。
            quantized (torch.Tensor): 量化后的向量，形状与输入相同。
            perplexity (torch.Tensor): 困惑度，用于评估量化器的性能。
            encodings (torch.Tensor): 编码后的向量，形状为[B, K]，其中K是代码书中的向量数量。
        """
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # BCHW --> BHWC
        input_shape = inputs.shape  # 假设[16, 32, 32, 64]

        flat_input = inputs.view(-1, self._embedding_dim)  # 将输入拉平，最后的维度与codebook向量维度一致，即[16*32*32, 64]

        # 计算输入向量与代码书向量之间的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1),
                     - 2*torch.matmul(flat_input, self._embedding.weight.t()))  # [16*32*32, K]

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # 获取最小距离的索引，[16*32*32, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)  # [16*32*32, K] 
        encodings.scatter_(1, encoding_indices, 1)  # [16*32*32, K]

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)  # [16, 32, 32, 64]

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # 重建损失
        q_laten_loss = F.mse_loss(quantized, inputs.detach())  # commitment损失
        loss = q_laten_loss + self._commitment_cost * e_latent_loss  # 总的损失

        quantized = inputs + (quantized - inputs).detach()  # 量化后的向量；实现量化向量的反向传播，同时保持输入的梯度不被量化操作影响
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # 困惑度

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
