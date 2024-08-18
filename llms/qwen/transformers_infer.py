
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
device = "cuda"

# 初始化模型
model = AutoModelForCausalLM.from_pretrained(
    "/root/share/new_models/qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("/root/share/new_models/qwen/Qwen2-7B-Instruct", padding_side="left")

# # 构建openai形式的history
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,  # 用于在输入中添加生成提示，该提示指向<|im_start|>assistant
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     c=1024,  # 控制生成的最大个数
# )  # 注意chat方法被generate方法取代，需要使用apply_chat_template将消息转换为模型能够理解的形式
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 批处理；批处理并不总能提高效率
message_batch = [
    [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
    # [{"role": "user", "content": "Hello!"}],  # 流式不支持批处理
]
text_batch = tokenizer.apply_chat_template(
    message_batch,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# generated_ids_batch = model.generate(
#     **model_inputs_batch,
#     max_new_tokens=512,
#     streamer=streamer
# )
# generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
# response_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
# # print(response_batch)  # 启动流式时，不用显示print就会自动输出

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 将打印的文本存储在一个队列中，以便下游应用程序作为迭代器来使用

# Use Thread to run generation in background
# Otherwise, the process is blocked until generation is complete
# and no streaming effect can be observed.
from threading import Thread
generation_kwargs = dict(model_inputs_batch, streamer=streamer, max_new_tokens=512)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

generated_text = ""
for new_text in streamer:
    generated_text += new_text
print(generated_text)