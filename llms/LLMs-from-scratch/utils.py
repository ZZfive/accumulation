import torch
import tiktoken
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size   

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


# 垃圾邮件微调数据集
class SpamDateset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: int = None, pad_token_id: int = 50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                              for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


'''
Instruct finetuning的数据集形式如下：
{'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
'''


# 模板添加函数
def format_input(entry: str, style: str = "alpaca") -> str:
    if style == "alpaca":
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text
    elif style == "phi":
        instruction_text = (
            f"<|user|>\n{entry['instruction']}"
        )

        input_text = f"\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text
    else:
        raise ValueError(f"{style} is not supported")
    

# Instruct finetuing数据集
class InstrcutionDataset(Dataset):
    def __init__(self, data, tokenizer, style, mask_instruction: bool = False):
        if style not in ["alpacha", "phi"]:
            raise ValueError(f"{style} is not supported")
        self.data = data
        self.mask_instruction = mask_instruction
        if self.mask_instruction:
            self.instruction_lengths = []  # 记录每个样本中instruction部分编码后的tokens长度

        self.encoded_texts = []
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            if style == "alpahca":
                response_text = f"\n\n### Response:\n{entry['output']}"
            elif style == "phi":
                response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

            if self.mask_instruction:
                instruction_length = len(tokenizer.encode(instruction_plus_input))
                self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        return self.instruction_lengths[index], self.encoded_texts[index] if self.mask_instruction else self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    

# collate函数，将dataset中的数据构建成便于模型训练的形式
def custom_collate_fn(batch: int,
                      pad_token_id: int = 50256,  # 样本结束后添加一个告诉模型何时停止
                      ignore_index: int = -100,  # 用于pad样本至max_length，使用torch.nn.functional.cross_entropy计算交叉熵损失时会忽略掉-100位置
                      allowed_max_length: int = None,  # 如果样本的最大长度超过了模型的最大长度，可用此参数进行截取
                      mask_instruction: bool = False,  # 构建target时是否将instruction部分mask，即对应位置设置为-100
                      device: str = "cpu"):
    batch_max_length = max(len(batch[i][1] if mask_instruction else batch[i]) for i in range(len(batch))) + 1 # 先找到当前batch的长度最大值，然后加1便于后续直接右移一位构建target

    inputs_lst, targets_lst = [], []

    for i in range(len(batch)):
        if mask_instruction:
            item = batch[i][1]
            instruction_length = batch[i][0]
        else:
            item = batch[i]
        
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))

        inputs = torch.tensor(padded[:-1])  # 训练时的输入
        targets = torch.tensor(padded[1:])  # 训练时的target

        # 将第一个pad_token后的所有替换为-100
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if mask_instruction:
            targets[:instruction_length-1] = -100  # Mask all input and instruction tokens in the targets

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor