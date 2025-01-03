import os
import io
import glob
import math
import tarfile
from typing import List, Tuple, Union
import torch
import torchaudio
import safetensors
from .configuration_whisper import WhisperVQConfig
from .modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts: List[Union[Tuple[torch.Tensor, int], str]]):
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.cuda()  # 将音频数据移动到GPU
            if sample_rate != 16000:  # 如果音频采样率不是16000，则进行重采样
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to('cuda')
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0  # 每30s提取一次音频
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length  # 计算步幅，1*2*4*160=1280
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)  # 提取mel谱图序列特征
            features = features.to(device="cuda")  # 包含input_features和attention_mask，尺寸例子分别为[1, 128, 120]和[1, 120]
            outputs = model(**features)  # 输出包含last_hidden_state[1, 15, 1280], quantized_token_ids[1, 15]
            speech_tokens = outputs.quantized_token_ids  # [1, 15]
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]  # [1, 60]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]  # [1, 15]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]  # 获取当前音频片段对应的原始音频索引
                # 使用attention_mask过滤出有效的标记
                # attention_mask[i].bool() 创建布尔掩码,只保留True位置的标记
                # speech_tokens[i][...] 使用布尔索引获取有效标记
                # .tolist() 将张量转换为Python列表
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)  # 将处理后的标记添加到对应原始音频的结果列表中
        return all_speech_tokens
