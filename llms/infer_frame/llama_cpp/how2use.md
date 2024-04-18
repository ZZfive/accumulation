本文档用于记录llama.cpp使用过程笔记

- [InternLM2](#InternLM2)
- [llama2](#llama2)
- [](#)
- [](#)
- [](#)


# InternLM2
&emsp;&emsp;最新版本(20240418)的llama.cpp应该是支持InternLM2，需要先将InternLM2模型转换为gguf，可按以下步骤部署
 - 本地下载InterLM2的HF模型，如internlm2-chat-7b
 - 进入llama.cpp的根目录: **cd your_path/llama.cpp**
 - 使用"convert-hf-to-gguf.py"脚本将internlm2-chat-7b转换为"ggml-model-f16.gguf": **python convert-hf-to-gguf.py your_path/internlm2-chat-7b**
 - 执行以下命令部署服务: **./main -m ~/Project/AIGC/internlm2-chat-7b/ggml-model-f16.gguf --temp 0.2 --top-p 0.9 --top-k 5 --repeat_penalty 1.1 -ngl 10 --color -ins**

# llama2