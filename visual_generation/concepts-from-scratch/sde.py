#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sde.py
@Time    :   2024/11/27 22:58:40
@Author  :   zzfive 
@Desc    :   手敲SDE简洁实现
'''
from copy import deepcopy
from typing import List, Callable, Tuple

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid


# 高斯随机特征的时间编码
class TimeEncoding(nn.Module):
    """
    时间编码层，用于将时间信息转换为特征向量。

    Attributes:
        w (nn.Parameter): 时间编码的参数，用于生成时间特征向量。
    """
    def __init__(self, embed_dim: int, scale: float = 30.) -> None:
        """
        初始化时间编码层。

        Args:
            embed_dim (int): 特征向量的维度。
            scale (float, optional): 时间编码的缩放因子。 Defaults to 30.
        """
        super().__init__()

        # 初始化时间编码参数，使用随机初始化，且不需要梯度更新
        self.w = nn.Parameter(torch.rand(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，生成时间特征向量。

        Args:
            x (torch.Tensor): 输入的时间信息。

        Returns:
            torch.Tensor: 生成的时间特征向量。
        """
        # 将时间信息与时间编码参数进行点乘，生成时间特征向量
        x_proj = x[:, None] * self.w[None, :] * 2 * np.pi
        # 使用正弦和余弦函数生成时间特征向量
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    密集连接层，用于将时间特征向量转换为特定维度的特征向量。

    Attributes:
        dense (nn.Linear): 密集连接层的线性变换。
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        初始化密集连接层。

        Args:
            input_dim (int): 输入特征向量的维度。
            output_dim (int): 输出特征向量的维度。
        """
        super().__init__()
        # 初始化线性变换层
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，进行线性变换。

        Args:
            x (torch.Tensor): 输入的时间特征向量。

        Returns:
            torch.Tensor: 输出的特征向量。
        """
        # 进行线性变换，并添加维度以适应卷积操作
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """
    评分网络，用于生成分数。

    Attributes:
        embed (nn.Sequential): 时间特征向量的嵌入层。
        conv1 (nn.Conv2d): 第一个卷积层。
        dense1 (Dense): 第一个密集连接层。
        gnorm1 (nn.GroupNorm): 第一个群归一化层。
        ... (nn.Module): 其他卷积层、密集连接层和群归一化层。
        tconv1 (nn.ConvTranspose2d): 第一个上采样卷积层。
        ... (nn.Module): 其他上采样卷积层和群归一化层。
        act (Callable): 激活函数。
        marginal_prob_std (Callable): 边际概率标准差函数。
    """
    def __init__(self, marginal_prob_std: Callable, input_channel: int = 1,
                 channels: List[int] = [32, 64, 128, 256],
                 embed_dim : int = 256) -> None:
        """
        初始化评分网络。

        Args:
            marginal_prob_std (Callable): 边际概率标准差函数。
            input_channel (int): 初始输入数据的通道维度，默认为1是后续使用MNIST数据集训练，其是只有灰度通道的一维图片。
            channels (List[int], optional): 卷积层的通道数列表。 Defaults to [32, 64, 128, 256].
            embed_dim (int, optional): 时间特征向量的嵌入维度。 Defaults to 256.
        """
        super().__init__()
        # 初始化时间特征向量的嵌入层
        self.embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        
        # 初始化卷积层、密集连接层和群归一化层
        self.conv1 = nn.Conv2d(input_channel, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # 初始化上采样卷积层和群归一化层
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], input_channel, 3, stride=1)

        # 定义激活函数
        self.act = lambda x: x * torch.sigmoid(x)
        # 保存边际概率标准差函数
        self.marginal_prob_std = marginal_prob_std
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播，生成分数。

        Args:
            x (torch.Tensor): 输入的特征向量。
            t (torch.Tensor): 时间信息。

        Returns:
            torch.Tensor: 生成的分数。
        """
        # 生成时间特征向量
        embed = self.act(self.embed(t))

        # 进行卷积操作和上采样操作
        h1 = self.conv1(x)
        h1 += self.dense1(embed)  # 注入时间特征
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv1(h1)
        h2 += self.dense1(embed)
        h2 = self.gnorm1(h2)
        h2 = self.act(h2)
        h3 = self.conv1(h2)
        h3 += self.dense1(embed)
        h3 = self.gnorm1(h3)
        h3 = self.act(h3)
        h4 = self.conv1(h3)
        h4 += self.dense1(embed)
        h4 = self.gnorm1(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h) 
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # 根据边际概率标准差进行标准化
        h = h / self.marginal_prob_std(t)[:, None, None, None]

        return h


'''
上述为主要模型架构定义，以下为SDE实际应用相关代码，以下为SDE定义为dx = \sigma^tdw时的代码实例
'''
def marginal_prob_std(t: float, sigma: float = 25.0, device: str = "cpu") -> torch.Tensor:
    """计算任意t时刻的扰动后条件高斯分布的标准差"""
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2*t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t: float, sigma: float = 25.0, device: str = "cpu") -> torch.Tensor:
    """计算任意t时刻的扩散系数，本例定义的SDE没有漂移系数"""
    return torch.tensor(sigma**t, device=device)


def loss_fn(score_model: nn.Module, x: torch.Tensor, marginal_prob_std: Callable, eps: float = 1e-5) -> torch.Tensor:
    """
    计算模型的损失函数，用于训练SDE模型。

    Args:
        model: SDE模型实例。
        x: 输入数据，形状为（批次大小，通道数，高度，宽度）。
        marginal_prob_std: 计算边际概率标准差的函数。
        eps: 一个小的正数，用于避免除以零的错误。 Defaults to 1e-5.

    Returns:
        torch.Tensor: 模型的损失值。
    """
    # 生成随机时间点，确保时间点在eps到1-eps之间，即从[0.00001, 0.9999]之间的随机均匀采样 浮点数
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # 基于重参数采样技巧采样出分布p_t(x)的一个随机样本，即扰动数据perturbed_x
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)  # 等价于sigma
    perturbed_x = x + z * std[:, None, None, None]

    # 使用模型对扰动后的输入数据进行评分，即预测分数
    score = score_model(perturbed_x, random_t)

    # 计算损失，损失函数为评分与随机噪声的平方和的均值
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss


class EMA(nn.Module):
    """
    实现指数移动平均（EMA）更新机制，用于模型参数的平滑更新。
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = None) -> None:
        """
        初始化EMA模块。

        Args:
            model (nn.Module): 需要进行EMA更新的模型实例。
            decay (float, optional): EMA更新的衰减率。 Defaults to 0.9999.
            device (str, optional): 模型所在的设备，用于设备转换。 Defaults to None.
        """
        super().__init__()

        # 深拷贝模型，确保EMA模块的模型参数独立于原始模型
        self.module = deepcopy(model)
        # 将模型设置为评估模式，避免在EMA更新过程中进行梯度计算
        self.module.eval()
        # 初始化衰减率
        self.decay = decay
        # 初始化设备
        self.device = device
        # 如果指定了设备，则将模型转移到该设备上
        if self.device is not None:
            self.module.to(device=device)
    
    def _update(self, model: nn.Module, update_fn: Callable) -> None:
        """
        更新EMA模块的参数。

        Args:
            model (nn.Module): 用于更新EMA模块参数的模型实例。
            update_fn (Callable): 更新函数，用于计算EMA参数的更新值。
        """
        with torch.no_grad():
            # 遍历EMA模块和模型的参数，进行更新
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                # 如果指定了设备，则将模型参数转移到该设备上
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                # 使用更新函数更新EMA参数
                ema_v.copy_(update_fn(ema_v, model_v))
    
    def update(self, model: nn.Module) -> None:
        """
        使用默认的EMA更新函数更新模型参数。

        Args:
            model (nn.Module): 用于更新EMA模块参数的模型实例。
        """
        # 调用内部更新函数，使用默认的EMA更新函数
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
    
    def set(self, model: nn.Module) -> None:
        """
        直接设置EMA模块的参数为模型参数，不进行EMA更新。

        Args:
            model (nn.Module): 用于设置EMA模块参数的模型实例。
        """
        # 调用内部更新函数，直接设置EMA参数为模型参数
        self._update(model, update_fn=lambda e, m: m)


'''
当SDE定义为dx = \sigma^tdw，则逆SDE为dx=-\sigma^{2t} s_\theta(x,t)dt+\sigma^td\bar{w}；基于一些数值解法就能采样
当使用欧拉数值采样时，用$\Delta_t$替代$dt$，用$z \sim N(0,g^2(t)\Delta_tI)$替代$d\omega$，
对应得到x_{t-\Delta{t}} = x_t + \sigma^{2t} s_\theta(x,t)\Delta{t}+\sigma^t\sqrt{\Delta{t}}z_t
'''
def euler_sampler(score_model: nn.Module, size: Tuple[int, int], marginal_prob_std: Callable, diffusion_coeff: Callable,
                  batch_size: int = 64, num_steps: int = 500, device: str = "cuda", eps: float = 1e-3) -> torch.Tensor:
    # 定义时间为1时，即先验分布中的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, size[0], size[1], device=device) * marginal_prob_std(t)[:, None, None, None]

    # 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # 根据欧拉算法求解逆SDE
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    
    # 返回最后一步的期望值作为生成的样本
    return mean_x


'''
除了单纯使用数值解法采样为，还可以将数值解法和模型解法结合在一起采样，如将欧拉采样和朗之万动力学采样结合，先使用欧拉采样获取一个时刻的预测值，
再使用朗之万动力学采样精炼，前者称为predictor，后者称为corrector，故此种采样简称为PC-Sampler
'''
def pc_sampler(score_model: nn.Module, size: Tuple[int, int], marginal_prob_std: Callable, diffusion_coeff: Callable, c_iters: int = 10,
               batch_size: int = 64, num_steps: int = 500, snr: float = 0.16, device: str = "cuda", eps: float = 1e-3) -> torch.Tensor:
    # 定义时间为1时，即先验分布中的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, size[0], size[1], device=device) * marginal_prob_std(t)[:, None, None, None]

    # 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # 嵌套循环采样，外层遍历所有逆SDE数值求解，内层执行朗之万动力学采样
    x = init_x
    with torch.no_grad():
          for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            # 朗之万采样
            for _ in range(c_iters):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.rand_like(x)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    
    # 返回最后一步的期望值作为生成的样本
    return mean_x


'''
对于所有扩散过程，在求解逆SDE时，存在一个相应的确定性过程，其轨迹与SDE具有相同的边缘概率密度，即逆向求解的结果和逆SDE求解出来的样本服从相同分布。这个确定性过程称为概率流常微分方程：dx=[f(x,t)-\frac{1}{2}g(t)^2\nabla_x \log p_t(x)]dt
当SDE定义为dx = \sigma^tdw时，对应逆ODE公式为dx=-\frac{1}{2}\sigma^{2t} s_\theta(x,t)dt
'''
def ode_sampler(score_model: nn.Module, size: Tuple[int, int], marginal_prob_std: Callable, diffusion_coeff: Callable,
                batch_size: int = 64, atol: float = 1e-5, rtol: float = 1e-5, device: str = "cuda", eps: float = 1e-3,
                z: torch.Tensor = None) -> torch.Tensor:
    # 定义时间为1时，即先验分布中的随机样本
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(batch_size, 1, size[0], size[1], device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
    
    shape = init_x.shape

    # 定义分数预测函数和常微分函数
    def score_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return 0.5 * (g**2) * score_eval_wrapper(x, time_steps)
    
    # 调用常微分求解算子解t=teps时刻的值，即预测的样本
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method="RK45")

    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x



if  __name__ == "__main__":
    ## 基于MNIST数据集训练
    # 设置设备为cuda
    device = "cuda"
    # 使用DataParallel将模型放在多个GPU上
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std))
    # 将模型移动到设备上
    score_model.to(device)

    # 设置训练的轮数
    n_epochs = 50
    # 设置批次大小
    batch_size = 32
    # 设置学习率
    lr = 1e-4

    # 加载MNIST数据集
    dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建优化器
    optimizer = Adam(score_model.parameters(), lr=lr)
    
    # 创建EMA模型
    ema_model = EMA(score_model)

    # 开始训练
    for epoch in range(n_epochs):
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            # 将数据移动到设备上
            x = x.to(device)
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 计算损失
            loss = loss_fn(score_model, x, marginal_prob_std)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新EMA模型
            ema_model.update(score_model)
            # 计算平均损失
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
    
    # 打印平均损失
    print("Averagr ScoreMatching Loss: {:5f}".format(avg_loss / num_items))
    # 保存模型参数
    torch.save({"score_model": score_model.state_dict(),
                "ema_model": ema_model.module.state_dict()},
                "ckpt.pth")
    
    # 采样
    sampler_batch_szie = 64
    sampler = pc_sampler  # ["euler_sampler", "pc_sampler", "ode_sampler"]三者之一
    
    samples = sampler(score_model=score_model,
                      batch_size=sampler_batch_szie,
                      device=device)
    
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sampler_batch_szie)))

    plt.figure(figsize=(6, 6))
    plt.axes("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()