#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   quick_start.py
@Time    :   2024/05/01 21:16:52
@Author  :   zzfive 
@Email   :   zhangte425922@outlook.com
'''

from vllm import LLM, SamplingParams

# 离线批量推理
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)  # SamplingParms用于配置采样过程参数

# LLM是vllm中运行离线推理的主要类
llm = LLM(model="/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b", dtype='bfloat16',
          trust_remote_code=True,  # 加载本地模型时可能会需要设置
          gpu_memory_utilization=0.95,  # 控制显存使用率，需要考虑显存大小设置
          enforce_eager=True,  # 使用本地的internlm2-chat-7b时，若不设置enforce_eager=True会报错
          max_model_len=1024)

# 采样生成
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


'''
vllm可以以api的形式部署兼容openai API格式的api服务
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m  # 执行部署的模型
    --chat-template ./examples/template_chatml.jinja  # 默认使用存储在tokenizer中的预定义chat模板，可以使用此参数进行设置

以下服务的启动命令与上述使用LLM构建离线推理引擎中设置一致
python -m vllm.entrypoints.openai.api_server
    --model /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b
    --trust-remote-code
    --enforce-eager
    --gpu-memory-utilization 0.95
    --max-model-len 1024
'''

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# OpenAI格式的completions形式的api调用
completion = client.completions.create(model="facebook/opt-125m",
                                      prompt="San Francisco is a")
print("Completion result:", completion)

# OpenAI格式的chat形式的api调用
chat_response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)


# from vllm.entrypoints.openai import api_server
# from vllm.entrypoints import api_server