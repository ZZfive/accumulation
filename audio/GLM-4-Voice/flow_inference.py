import torch
import torchaudio
import numpy as np
import re
from hyperpyyaml import load_hyperpyyaml
import uuid
from collections import defaultdict


def fade_in_out(fade_in_mel, fade_out_mel, window):  # 平滑拼接两段mel谱图
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class AudioDecoder:
    """音频解码器类
    
    主要功能:
    1. 将token序列解码为mel谱图
    2. 将mel谱图转换为音频波形
    
    关键参数:
    - token_overlap_len: token重叠长度，控制解码的连续性
    - mel_overlap_len: mel谱图重叠长度，用于平滑拼接
    - mel_cache_len: mel谱图缓存长度，影响生成质量
    - source_cache_len: 源音频缓存长度，确保音频连续
    
    缓存机制:
    - mel_overlap_dict: 存储mel谱图重叠区域
    - hift_cache_dict: 存储HiFi-GAN状态
    
    窗口函数:
    - mel_window: mel谱图重叠区域的Hamming窗口
    - speech_window: 音频重叠区域的Hamming窗口
    """
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda"):
        self.device = device

        with open(config_path, 'r') as f:
            self.scratch_configs = load_hyperpyyaml(f)  # 可直接基于配置文件实例化对象或函数

        # Load models
        self.flow = self.scratch_configs['flow']  # flow decoder
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device))
        self.hift = self.scratch_configs['hift']  # vocoder
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate  # input_frame_rate是输入的音频帧率
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5  # token级别重叠长度
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)  # mel谱图重叠长度
        self.mel_window = np.hamming(2 * self.mel_overlap_len)  # Hamming窗口是一种重要的窗函数，主要用于平滑信号过度和减少频谱泄露
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)  # mel谱图到音频样本的转换
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False):
        """
        token: 当前带解码的speech tokens
        uuid: 
        prompt_token: 架构与cosyvoice一致，此对应于cosyvoice中的参考音频的speech token序列，此处是所有累计预测出的speech token，初始相当于一个空序列
        prompt_feat: 架构与cosyvoice一致，此对应于cosyvoice中的参考音频的mel谱图序列，此处是所有累计预测出的mel谱图，初始相当于一个空序列
        embedding: 可用于设置输出音频的音色embeddings
        finalize: 是否是最后一个待解码的speech tokens序列
        """
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device))  # [1, 80, 172]，通过flow decoder预测出mel谱图

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)  # 平滑拼接两段mel谱图
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)  # 将之前预测出的mel谱图和当前预测出的mel谱图拼接

        else:
            hift_cache_source = torch.zeros(1, 1, 0)  # 如果之前没有预测出的mel谱图，则初始化hift_cache_source为0
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:  # 如果当前不是最后一个待解码的speech tokens序列，则进行以下操作
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]  # 将当前预测出的mel谱图的最后一段保存到mel_overlap_dict中，mel_overlap_len为34，即[1, 80, 34]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]  # 将当前预测出的mel谱图的最后一段截取掉, [1, 80, 138]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)  # 通过hift decoder预测出音频和源音频, 预测的两个结果shape分别为[1, 35328]和[1, 1, 35328]

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],  # 将当前预测出的mel谱图的最后一段保存到hift_cache_dict中, [1]
                                          'source': tts_source[:, :, -self.source_cache_len:],  # 将当前预测出的源音频的最后一段保存到hift_cache_dict中, [1, 1, 256]
                                          'speech': tts_speech[:, -self.source_cache_len:]}  # 将当前预测出的音频的最后一段保存到hift_cache_dict中, [1, 256]
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, :-self.source_cache_len]  # 将当前预测出的音频的最后一段截取掉, [1, 35072]

        else:  # 如果当前是最后一个待解码的speech tokens序列，则进行以下操作
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)  # 通过hift decoder预测出音频和源音频
            del self.hift_cache_dict[uuid]  # 删除hift_cache_dict中的当前预测出的mel谱图和源音频
            del self.mel_overlap_dict[uuid]  # 删除mel_overlap_dict中的当前预测出的mel谱图
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speechs = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx:idx + block_size]

            print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech
            print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech.cpu()

