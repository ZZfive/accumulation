#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   api_client.py
@Time    :   2024/05/02 08:38:32
@Author  :   zzfive 
@Email   :   zhangte425922@outlook.com
'''

"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


# 清除终端或控制台输出中的特定行数；如果调用 clear_line()，它将清除一行。如果你调用 clear_line(2)，它将清除两行
def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'  # 此 ANSI 转义码将光标向上移动一行
    LINE_CLEAR = '\x1b[2K'  # 此 ANSI 转义码清除当前行的内容
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)  # 根据传入的参数n，循环 n 次，每次打印移动光标和清除行内容的转义码


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": True,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output  # 以生成器的形式返回


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"  # 与openai方式启动服务时不同，vllm.entrypoints.api_server启动后只有generate一个endpoint
    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    if stream:  # 流式输出
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)  # 为了体现出流式的形式，在每次输出新token之前，把之前的对应行的输出删除掉，再输出完整的内容，token就像一个一个的输出一样
            num_printed_lines = 0
            for i, line in enumerate(h):  # 因为请求时指定了n，每次都会有n个输出
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)