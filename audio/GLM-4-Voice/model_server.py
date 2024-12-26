"""
A model worker with transformers libs executes the model.

Run BF16 inference with:

python model_server.py --host localhost --model-path THUDM/glm-4-voice-9b --port 10000 --dtype bfloat16 --device cuda:0

Run Int4 inference with:

python model_server.py --host localhost --model-path THUDM/glm-4-voice-9b --port 10000 --dtype int4 --device cuda:0

"""
import argparse
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
import torch
import uvicorn

from threading import Thread
from queue import Queue


class TokenStreamer(BaseStreamer):  # 继承自BaseStreamer的流式处理器，用于生成的token流式输出
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt  # 是否跳过prompt

        # variables used in the streaming process
        self.token_queue = Queue()  # 用于存储生成的token，作为生成线程和主线程之间的中间存储，queue是线程安全的，可以安全地在不同线程间传递数据
        self.stop_signal = None  # 停止信号
        self.next_tokens_are_prompt = True  # 下一个token是否是prompt
        self.timeout = timeout  # 超时时间

    def put(self, value):
        """
        在生成线程中被调用
        model.generate() -> transformers内部 -> put()
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)  # 将停止信号添加到token队列中

    # 实现迭代器接口，允许流式获取token
    def __iter__(self):
        return self  # 返回迭代器

    def __next__(self):
        """
        在主线程中被调用
        for token in streamer -> __next__()
        """
        value = self.token_queue.get(timeout=self.timeout)  # 从token队列中获取token，如果队列为空会阻塞等待
        if value == self.stop_signal:
            raise StopIteration()  # 如果获取到停止信号，则抛出StopIteration异常
        else:
            return value


"""
生成线程(model.generate)  -->  Streamer(Queue)  -->  主线程(yield给客户端)
     |                          |                        |
     |                          |                        |
生成token                    存储token                获取token
     ↓                          ↓                        ↓
调用streamer.put()        token进入队列          从队列获取token并yield
"""
class ModelWorker:
    def __init__(self, model_path, dtype="bfloat16", device='cuda'):
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map={"": 0}
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt")  # 对输入进行编码
        inputs = inputs.to(self.device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(
            target=model.generate,  # transformers实现的generate实现中会自动调用streamer.put()和streamer.end()
            kwargs=dict(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                streamer=streamer
            )
        )
        thread.start()  # 启动线程
        for token_id in streamer:  # 会调用streamer的__next__()方法
            # 流式返回生成的token
            yield (json.dumps({"token_id": token_id, "error_code": 0}) + "\n").encode()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret) + "\n").encode()


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = worker.generate_stream_gate(params)
    return StreamingResponse(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    args = parser.parse_args()

    worker = ModelWorker(args.model_path, args.dtype, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")