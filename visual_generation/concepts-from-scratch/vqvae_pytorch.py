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
from typing import Tuple

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

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # commitment损失
        q_laten_loss = F.mse_loss(quantized, inputs.detach())  # codebook损失
        loss = q_laten_loss + self._commitment_cost * e_latent_loss  # 总的损失

        quantized = inputs + (quantized - inputs).detach()  # 量化后的向量；实现量化向量的反向传播，同时保持输入的梯度不被量化操作影响
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # 困惑度

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


# EMA版本
class VectorQuantizerEMA(nn.Module):
    """
    定义一个使用指数移动平均（EMA）更新代码书的向量量化器（Vector Quantizer）。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float,
                 decay: float, epsilon: float = 1e-5) -> None:
        """
        初始化向量量化器。

        Args:
            num_embeddings (int): 代码书中向量的数量，即K。
            embedding_dim (int): 每个向量的维度，即D。
            commitment_cost (float): 损失函数中的commitment loss的系数，用于防止代码书中的向量无限制增长。
            decay (float): 指数移动平均的衰减率，用于控制代码书向量的更新速度。
            epsilon (float, optional): 一个小的正数，用于避免除以零的错误。 Defaults to 1e-5.
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # 初始化代码书，使用nn.Embedding模块来存储代码书中的向量。
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 初始化代码书向量的权重，使用标准正态分布初始化。
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        # 注册一个缓冲区来存储每个代码书向量的聚类大小，表示每个codebook向量倍使用的频率
        self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings))
        # 初始化一个参数来存储代码书向量的EMA权重。
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        # 初始化EMA权重，使用标准正态分布。
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1)
                     - 2*torch.matmul(flat_input, self._embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # 使用EMA更新embedding向量
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)  # 计算当前所有codebook向量的总聚类大小n，用于后续的归一化
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n )
            dw = torch.matmul(encodings.t(), flat_input)  # codebook向量对于输入的反向传播梯度
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)  # 更新codebook的EMA权重
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))  # 更新codebook权重

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        super().__init__()
        self._num_residual_alyers = num_residual_layers
        self._lays = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_alyers
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self._num_residual_alyers):
            x = self._lays[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv2 = nn.Conv2d(in_channels=num_hiddens//2,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=in_channels,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = F.relu(x)

        x = self._conv2(x)
        x = F.relu(x)

        x = self._conv3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQVAEModel(nn.Module):
    def __init__(self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int,
                 num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float = 0) -> None:
        super().__init__()
        self._encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1, stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                             commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers,
                                num_residual_hiddens)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity