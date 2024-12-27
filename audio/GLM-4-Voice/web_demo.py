import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
from argparse import ArgumentParser

import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder


sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token

import gradio as gr
import torch

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

from flow_inference import AudioDecoder
from audio_process import AudioStreamProcessor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type= str, default="THUDM/glm-4-voice-tokenizer")
    args = parser.parse_args()

    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    glm_tokenizer = None
    device = "cuda"
    audio_decoder: AudioDecoder = None
    whisper_model, feature_extractor = None, None


    def initialize_fn():
        global audio_decoder, feature_extractor, whisper_model, glm_model, glm_tokenizer
        if audio_decoder is not None:
            return

        # GLM
        glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        # Flow & Hift，架构与CosyVoice一致，只是vocab size等少数参数设置不同
        audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                     hift_ckpt_path=hift_checkpoint,
                                     device=device)

        # Speech tokenizer
        whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)  # from_pretrained方法会自动从指定的路径下读取config.json文件初始化模型后在加载模型权重
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)  # 从指定的路径下读取preprocessor_config.json文件初始化特征提取器


    def clear_fn():
        return [], [], '', '', '', None, None


    def inference_fn(
            temperature: float,
            top_p: float,
            max_new_token: int,
            input_mode,
            audio_path: str | None,
            input_text: str | None,
            history: list[dict],
            previous_input_tokens: str,
            previous_completion_tokens: str,
    ):

        if input_mode == "audio":
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [audio_path]
            )[0]  # 提取音频的token，返回一个列表，列表中的每个元素是一个token id，就像文本分词器返回的token id
            if len(audio_tokens) == 0:
                raise gr.Error("No audio tokens extracted")
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])  # 将token id转换为token字符串
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"  # 添加音频内容的开始和结束标记
            user_input = audio_tokens
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        else:
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            user_input = input_text
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."


        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"  # 添加系统提示
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"  # 添加用户输入和助手提示

        with torch.no_grad():
            response = requests.post(
                "http://localhost:10000/generate_stream",
                data=json.dumps({
                    "prompt": inputs,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_token,
                }),
                stream=True
            )  # 发送请求并获取流式响应
            text_tokens, audio_tokens = [], []
            audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>') # 152353
            end_token_id = glm_tokenizer.convert_tokens_to_ids('<|user|>')  # 151336
            complete_tokens = []
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)  # 初始化prompt_speech_feat
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
            this_uuid = str(uuid.uuid4())
            tts_speechs = []
            tts_mels = []
            prev_mel = None
            is_finalize = False
            block_size_list =  [25,50,100,150,200]
            block_size_idx = 0
            block_size = block_size_list[block_size_idx]
            audio_processor = AudioStreamProcessor()
            for chunk in response.iter_lines():
                token_id = json.loads(chunk)["token_id"]  # 从流式响应中提取token id
                if token_id == end_token_id:
                    is_finalize = True
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):  # 如果音频token数量大于等于block_size，或者已经到达结束标记，则进行TTS
                    if block_size_idx < len(block_size_list) - 1:
                        block_size_idx += 1
                        block_size = block_size_list[block_size_idx]  # 更新block_size
                    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)  # 将音频token id转换为张量， 如[1, 25]

                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)  # 初始时prompt_speech_feat为0，后续将tts_mels连接起来

                    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(device),  # 将之前预测预测的speech token作为后续预测的条件
                                                                  prompt_feat=prompt_speech_feat.to(device),  # 将之前预测出的mel谱图序列也所谓后续预测的条件
                                                                  finalize=is_finalize)  # 使用flow的audio decoder模型将speech token转换为音频和mel谱图
                    prev_mel = tts_mel  # 更新prev_mel

                    audio_bytes = audio_processor.process(tts_speech.clone().cpu().numpy()[0], last=is_finalize)  # 处理音频

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    if audio_bytes:  # 如果音频数据不为空，则返回历史、输入、空字符串、空字符串、音频数据、None
                        yield history, inputs, '', '', audio_bytes, None
                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)  # 更新flow_prompt_speech_token
                    audio_tokens = []  # 每次解码预测一次mel谱图后回清空audio_tokens，方式重复包含之前的audio tokens
                if not is_finalize:
                    complete_tokens.append(token_id)  # 所有预测出的token id序列
                    if token_id >= audio_offset:  # 如果token id大于音频偏移量，则认为是音频token
                        audio_tokens.append(token_id - audio_offset)  # 将音频token id转换为音频token
                    else:
                        text_tokens.append(token_id)  # 否则认为是文本token
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()  # 将所有音频片段连接起来，并转换为CPU上的张量
        complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)  # 将所有文本token连接起来，并转换为文本
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        history.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
        history.append({"role": "assistant", "content": glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)})
        yield history, inputs, complete_text, '', None, (22050, tts_speech.numpy())


    def update_input_interface(input_mode):
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]


    # Create the Gradio interface
    with gr.Blocks(title="GLM-4-Voice Demo", fill_height=True) as demo:
        with gr.Row():
            temperature = gr.Number(
                label="Temperature",
                value=0.2
            )

            top_p = gr.Number(
                label="Top p",
                value=0.8
            )

            max_new_token = gr.Number(
                label="Max new tokens",
                value=2000,
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(["audio", "text"], label="Input Mode", value="audio")
                audio = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                text_input = gr.Textbox(label="Input text", placeholder="Enter your text here...", lines=2, visible=False)

            with gr.Column():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                output_audio = gr.Audio(label="Play", streaming=True,
                                        autoplay=True, show_download_button=False)
                complete_audio = gr.Audio(label="Last Output Audio (If Any)", show_download_button=True)



        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(
                label=f"Input Tokens",
                interactive=False,
            )

            completion_tokens = gr.Textbox(
                label=f"Completion Tokens",
                interactive=False,
            )

        detailed_error = gr.Textbox(
            label=f"Detailed Error",
            interactive=False,
        )

        history_state = gr.State([])

        respond = submit_btn.click(
            inference_fn,
            inputs=[
                temperature,
                top_p,
                max_new_token,
                input_mode,
                audio,
                text_input,
                history_state,
                input_tokens,
                completion_tokens,
            ],
            outputs=[history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]
        )

        respond.then(lambda s: s, [history_state], chatbot)

        reset_btn.click(clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio])
        input_mode.input(clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]).then(update_input_interface, inputs=[input_mode], outputs=[audio, text_input])

    initialize_fn()
    # Launch the interface
    demo.launch(
        server_port=args.port,
        server_name=args.host
    )
