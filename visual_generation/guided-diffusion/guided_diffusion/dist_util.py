"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    设置分布式进程组。
    """
    # 检查分布式进程是否已初始化
    if dist.is_initialized():
        return
    # 根据MPI进程排名分配GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    # 获取MPI通信世界
    comm = MPI.COMM_WORLD
    # 根据是否有CUDA设备选择分布式后端
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    # 根据后端选择主机名
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    # 广播主机名到所有进程
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # 设置进程排名和世界大小环境变量
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    # 广播空闲端口到所有进程
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    # 使用环境变量初始化分布式进程组
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    在MPI排名之间不重复地加载PyTorch文件。
    这个函数旨在处理分布式环境中PyTorch状态字典的加载。
    它确保文件只被根进程加载一次，然后广播到所有其他进程中，
    避免了重复的文件访问，提高了效率。
    """
    # 定义用于广播数据的块大小。由于MPI的限制，这被设置为一个相对较小的大小限制。
    chunk_size = 2 ** 30
    # 检查当前进程是否是根进程（排名0）。
    if MPI.COMM_WORLD.Get_rank() == 0:
        # 使用BlobFile以二进制读模式打开文件。
        with bf.BlobFile(path, "rb") as f:
            # 读取整个文件内容。
            data = f.read()
        # 计算需要广播数据的块数。
        num_chunks = len(data) // chunk_size
        # 如果有余数，这意味着最后一个块将小于块大小。
        if len(data) % chunk_size:
            num_chunks += 1
        # 广播块数到所有进程。
        MPI.COMM_WORLD.bcast(num_chunks)
        # 广播每个数据块到所有进程。
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        # 如果这不是根进程，从根进程接收块数。
        num_chunks = MPI.COMM_WORLD.bcast(None)
        # 初始化一个空的字节对象来累积接收到的数据块。
        data = bytes()
        # 从根进程接收每个数据块，并累积它们。
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    # 将累积的数据加载到PyTorch状态字典中并返回。
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
