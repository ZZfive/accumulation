import datetime

from vllm import LLM, SamplingParams

# 离线批量推理
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/root/models/internlm2-chat-7b", dtype='bfloat16',
          trust_remote_code=True,  # 必须
          gpu_memory_utilization=0.95,  # 必须
          enforce_eager=True)  # 使用本地的internlm2-chat-7b时，若不设置enforce_eager=True会报错

# 采样生成
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# # 计算words/s
# # warmup
# inp = "hello"
# for i in range(5):
#     print("Warm up...[{}/5]".format(i+1))
#     response = llm.generate([inp], sampling_params)

# # test speed
# inp = "请介绍一下你自己。"
# times = 10
# total_words = 0
# start_time = datetime.datetime.now()
# for i in range(times):
#     response = llm.generate([inp], sampling_params)
#     total_words += len(response[0].outputs[0].text)
# end_time = datetime.datetime.now()

# delta_time = end_time - start_time
# delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
# speed = total_words / delta_time
# print("Speed: {:.3f} words/s".format(speed))