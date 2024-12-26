import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import io

# Split audio stream at silence points to prevent playback stuttering issues
# caused by AAC encoder frame padding when streaming audio through Gradio audio components.
# 用于处理音频流，特别是在静音点处分割音频，以防止在通过 Gradio 音频组件流式传输音频时出现播放卡顿问题
class AudioStreamProcessor:
    def __init__(self, sr=22050, min_silence_duration=0.1, threshold_db=-40):
        self.sr = sr  # 采样率
        self.min_silence_duration = min_silence_duration  # 最小静音持续时间
        self.threshold_db = threshold_db  # 阈值
        self.buffer = np.array([])  # 缓冲区
    
    def process(self, audio_data, last=False):
        """
        Add audio data and process it
        params:
            audio_data: audio data in numpy array
            last: whether this is the last chunk of data
        returns:
            Processed audio data, returns None if no split point is found
        """

        # Add new data to buffer
        self.buffer = np.concatenate([self.buffer, audio_data]) if len(self.buffer) > 0 else audio_data  # 将新数据添加到缓冲区
        
        if last:  # 如果这是最后一个数据块，则返回整个缓冲区
            result = self.buffer
            self.buffer = np.array([])
            return self._to_wav_bytes(result)  # 将缓冲区转换为WAV格式
            
        # Find silence boundary
        split_point = self._find_silence_boundary(self.buffer)  # 找到缓冲区中的静音段起始点
        
        if split_point is not None:  # 如果找到了静音段起始点
            # Modified: Extend split point to the end of silence
            silence_end = self._find_silence_end(split_point)  # 找到缓冲区中的静音段结束点
            result = self.buffer[:silence_end]  # 将缓冲区中的静音段截断
            self.buffer = self.buffer[silence_end:]  # 将缓冲区中的静音段截断后的剩余部分赋值给缓冲区
            return self._to_wav_bytes(result)  # 将缓冲区转换为WAV格式
            
        return None
        
    def _find_silence_boundary(self, audio):  # 找到音频中静音段的起始点
        """
        Find the starting point of silence boundary in audio
        """
        # Convert audio to decibels
        db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)  # 将音频转换为分贝
        
        # Find points below threshold
        silence_points = np.where(db < self.threshold_db)[0]  # 找到低于阈值的点
        
        if len(silence_points) == 0:  # 如果找不到低于阈值的点，则返回None
            return None
            
        # Calculate minimum silence samples
        min_silence_samples = int(self.min_silence_duration * self.sr)  # 计算最小静音样本数
        
        # Search backwards for continuous silence segment starting point
        for i in range(len(silence_points) - min_silence_samples, -1, -1):
            if i < 0:  # 如果i小于0，则跳出循环
                break
            if np.all(np.diff(silence_points[i:i+min_silence_samples]) == 1):  # 检查连续静音段
                return silence_points[i]  # 返回静音段起始点
                
        return None
        
    def _find_silence_end(self, start_point):  # 找到音频中静音段的结束点
        """
        Find the end point of silence segment
        """
        db = librosa.amplitude_to_db(np.abs(self.buffer[start_point:]), ref=np.max)  # 将音频数据转换为分贝
        silence_points = np.where(db >= self.threshold_db)[0]  # 找到高于阈值的点
        
        if len(silence_points) == 0:  # 如果找不到高于阈值的点，则返回音频数据的长度
            return len(self.buffer)
            
        return start_point + silence_points[0]  # 返回静音段结束点
      
    def _to_wav_bytes(self, audio_data):  # 将音频数据转换为WAV格式
        """
        trans_to_wav_bytes
        """
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, self.sr, format='WAV')
        return wav_buffer.getvalue()
      
    
