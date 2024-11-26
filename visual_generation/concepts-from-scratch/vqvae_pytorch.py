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

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
    """
    残差块类，用于构建残差网络。

    Attributes:
        _block (nn.Sequential): 残差块的网络结构，包含两个卷积层和两个ReLU激活函数。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int) -> None:
        """
        初始化残差块。

        Args:
            in_channels (int): 输入通道数。
            num_hiddens (int): 隐藏层通道数。
            num_residual_hiddens (int): 残差连接的隐藏层通道数。
        """
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),  # 第一个ReLU激活函数
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),  # 第一个卷积层
            nn.ReLU(True),  # 第二个ReLU激活函数
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)  # 第二个卷积层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入的特征图。

        Returns:
            torch.Tensor: 输出的特征图，包括残差连接的结果。
        """
        return x + self._block(x)  # 残差连接


class ResidualStack(nn.Module):
    """
    残差栈类，用于构建多个残差块的堆叠。

    Attributes:
        _num_residual_layers (int): 残差块的数量。
        _layers (nn.ModuleList): 包含多个残差块的列表。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        """
        初始化残差栈。

        Args:
            in_channels (int): 输入通道数。
            num_hiddens (int): 隐藏层通道数。
            num_residual_layers (int): 残差块的数量。
            num_residual_hiddens (int): 残差连接的隐藏层通道数。
        """
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入的特征图。

        Returns:
            torch.Tensor: 输出的特征图，经过所有残差块的处理。
        """
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)  # 最后一个ReLU激活函数


class Encoder(nn.Module):
    """
    编码器类，用于将输入图像编码为特征向量。

    Attributes:
        _conv1 (nn.Conv2d): 第一个卷积层。
        _conv2 (nn.Conv2d): 第二个卷积层。
        _conv3 (nn.Conv2d): 第三个卷积层。
        _residual_stack (ResidualStack): 残差栈。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        """
        初始化编码器。

        Args:
            in_channels (int): 输入通道数。
            num_hiddens (int): 隐藏层通道数。
            num_residual_layers (int): 残差块的数量。
            num_residual_hiddens (int): 残差连接的隐藏层通道数。
        """
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2, padding=1)  # 下采样卷积层
        self._conv2 = nn.Conv2d(in_channels=num_hiddens//2,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1)  # 下采样卷积层
        self._conv3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)  # 普通卷积层
        self._residual_stack = ResidualStack(in_channels=in_channels,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入的图像特征图。

        Returns:
            torch.Tensor: 输出的特征向量，经过编码器的处理。
        """
        x = self._conv1(x)
        x = F.relu(x)  # ReLU激活函数

        x = self._conv2(x)
        x = F.relu(x)  # ReLU激活函数

        x = self._conv3(x)
        return self._residual_stack(x)  # 通过残差栈处理


class Decoder(nn.Module):
    """
    解码器类，用于将特征向量解码为图像。

    Attributes:
        _conv1 (nn.Conv2d): 第一个卷积层。
        _residual_stack (ResidualStack): 残差栈。
        _conv_trans_1 (nn.ConvTranspose2d): 第一个上采样卷积层。
        _conv_trans_2 (nn.ConvTranspose2d): 第二个上采样卷积层。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int) -> None:
        """
        初始化解码器。

        Args:
            in_channels (int): 输入通道数。
            num_hiddens (int): 隐藏层通道数。
            num_residual_layers (int): 残差块的数量。
            num_residual_hiddens (int): 残差连接的隐藏层通道数。
        """
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)  # 普通卷积层
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)  # 上采样卷积层
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)  # 上采样卷积层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入的特征向量。

        Returns:
            torch.Tensor: 输出的图像特征图，经过解码器的处理。
        """
        x = self._conv1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)  # ReLU激活函数

        return self._conv_trans_2(x)  # 最终的上采样卷积层


class VQVAEModel(nn.Module):
    """
    VQVAE模型类，用于实现矢量量化自编码器。

    Attributes:
        _encoder (Encoder): 编码器。
        _pre_vq_conv (nn.Conv2d): 量化前的卷积层。
        _vq_vae (VectorQuantizer or VectorQuantizerEMA): 矢量量化器。
        _decoder (Decoder): 解码器。
    """
    def __init__(self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int,
                 num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float = 0) -> None:
        """
        初始化VQVAE模型。

        Args:
            num_hiddens (int): 隐藏层通道数。
            num_residual_layers (int): 残差块的数量。
            num_residual_hiddens (int): 残差连接的隐藏层通道数。
            num_embeddings (int): 量化后的向量数量。
            embedding_dim (int): 量化后的向量维度。
            commitment_cost (float): 承诺成本，用于量化损失。
            decay (float, optional): 衰减率，用于EMA更新。 Defaults to 0.
        """
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
    

def show(img):
    """
    显示给定的图像。

    Args:
        img (torch.Tensor): 要显示的图像。

    Notes:
        这个函数将给定的图像转换为numpy数组，调整通道顺序以适应matplotlib的显示格式，
        并使用matplotlib显示图像。它还隐藏了坐标轴以获得更好的视觉效果。
    """
    npimg = img.numpy()  # 将torch.Tensor转换为numpy数组
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')  # 调整通道顺序并显示图像
    fig.axes.get_xaxis().set_visible(False)  # 隐藏x轴
    fig.axes.get_yaxis().set_visible(False)  # 隐藏y轴


if __name__ == "__main__":
    # 使用小型数据集训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的CUDA设备

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),  # 将数据转换为torch.Tensor
                                         transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))  # 数据标准化
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                       ]))
    
    data_variance = np.var(training_data.data / 255.0)  # 计算训练数据的方差
    
    batch_size = 256  # 设置批次大小
    num_training_updates = 15000  # 设置训练更新次数
    num_hiddens = 128  # 设置隐藏层通道数
    num_residual_hiddens = 32  # 设置残差连接的隐藏层通道数
    num_residual_layers = 2  # 设置残差块的数量
    embedding_dim = 64  # 设置量化后的向量维度
    num_embeddings = 512  # 设置量化后的向量数量
    commitment_cost = 0.25  # 设置承诺成本
    decay = 0.99  # 设置衰减率
    learning_rate = 1e-3  # 设置学习率

    training_loader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)  # 创建训练数据加载器
    validation_loader = DataLoader(validation_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)  # 创建验证数据加载器
    
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                       embedding_dim, commitment_cost, decay).to(device)  # 创建模型并移动到设备
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)  # 创建优化器
    
    model.train()  # 设置模型为训练模式
    train_res_recon_error = []  # 初始化重建误差列表
    train_res_perplexity = []  # 初始化困惑度列表

    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))  # 从训练加载器中获取数据
        data = data.to(device)  # 将数据移动到设备
        optimizer.zero_grad()  # 清空优化器的梯度

        vq_loss, data_recon, perplexity = model(data)  # 前向传播
        recon_error = F.mse_loss(data_recon, data) / data_variance  # 计算重建误差
        loss = recon_error + vq_loss  # 计算总损失
        loss.backward()  # 反向传播

        train_res_recon_error.append(recon_error.item())  # 记录重建误差
        train_res_perplexity.append(perplexity.item())  # 记录困惑度

        if (i+1) % 100 == 0:  # 每100次迭代打印一次
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))  # 打印最近100次的平均重建误差
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))  # 打印最近100次的平均困惑度
            print()

    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)  # 平滑重建误差
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)  # 平滑困惑度

    f = plt.figure(figsize=(16,8))  # 创建图形
    ax = f.add_subplot(1,2,1)  # 创建子图
    ax.plot(train_res_recon_error_smooth)  # 绘制平滑后的重建误差
    ax.set_yscale('log')  # 设置y轴为对数刻度
    ax.set_title('Smoothed NMSE.')  # 设置标题
    ax.set_xlabel('iteration')  # 设置x轴标签

    ax = f.add_subplot(1,2,2)  # 创建另一个子图
    ax.plot(train_res_perplexity_smooth)  # 绘制平滑后的困惑度
    ax.set_title('Smoothed Average codebook usage (perplexity).')  # 设置标题
    ax.set_xlabel('iteration')  # 设置x轴标签

    # 重建
    model.eval()  # 设置模型为评估模式

    (valid_originals, _) = next(iter(validation_loader))  # 从验证加载器中获取数据
    valid_originals = valid_originals.to(device)  # 将数据移动到设备

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))  # 量化前的卷积
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)  # 量化
    valid_reconstructions = model._decoder(valid_quantize)  # 解码重建

    # (train_originals, _) = next(iter(training_loader))
    # train_originals = train_originals.to(device)
    # _, train_reconstructions, _, _ = model._vq_vae(train_originals)

    show(make_grid(valid_reconstructions.cpu().data)+0.5)  # 显示重建的验证集图像
    show(make_grid(valid_originals.cpu()+0.5))  # 显示原始的验证集图像