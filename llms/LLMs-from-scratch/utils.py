import torch
import tiktoken
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def text2id(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def id2text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) ->str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


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
def custom_collate_fn(batch,
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


"""
DPO主要思想就是对于同一个prompt有两个不同的回答，可以记为chosen和rejected；在设置的偏好上，如帮助性、安全性上chosen的内容比rejected更符合人类的预期，如下所示：
{'instruction': 'Identify the correct spelling of the following word.',
 'input': 'Ocassion',
 'output': "The correct spelling is 'Occasion.'",
 'rejected': "The correct spelling is obviously 'Occasion.'",
 'chosen': "The correct spelling is 'Occasion.'"}

上述例子中instruction、input、output与SFT阶段中一致，而rejected、chosen可以人写或使用LLMs生成
"""


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, style):
        if style not in ["alpacha", "phi"]:
            raise ValueError(f"{style} is not supported")
        self.data = data

        self.encoded_texts = []
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            if style == "alpahca":
                rejected_response = f"\n\n### Response:\n{entry['rejected']}"
                chosen_response = f"\n\n### Response:\n{entry['chosen']}"
            elif style == "phi":
                rejected_response = f"\n<|assistant|>:\n{entry['rejected']}"
                chosen_response = f"\n<|assistant|>:\n{entry['chosen']}"
            
            instruction_plus_input_tokens = tokenizer.encode(instruction_plus_input)
            rejected_full_text = instruction_plus_input + rejected_response
            chosen_full_text = instruction_plus_input + chosen_response
            rejected_full_tokens = tokenizer.encode(rejected_full_text)
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            self.encoded_texts.append({
                "prompt": instruction_plus_input_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    

def custom_collate_fn_dpo(batch,
                          pad_token_id: int = 50256,
                          allowed_max_length: int = None,
                          mask_promt_tokens: bool = True,
                          device: str = "cpu"):
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "chosen_mask": [],
        "rejected_mask": []
    }  # make use of this mask to ignore prompt and padding tokens when computing the DPO loss

    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) for item in batch) + 1
            max_length_common = max(max_length_common, current_max)

    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(max_length_common).bool()

            mask[len(sequence):] = False  # Set mask for all padding tokens to False???为什么和上述保留了一个pad_token用于告诉模型何时停止不一样

            if mask_promt_tokens:
                mask[:prompt.shape[0]+2] = False  # Set mask for all input tokens to False; +2 sets the 2 newline ("\n") tokens before "### Response" to False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(torch.tensor(mask))

    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])

        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        batch[key] = tensor_stack.to(device)

    return batch_data