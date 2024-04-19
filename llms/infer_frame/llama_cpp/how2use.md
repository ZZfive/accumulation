**本文档用于记录llama.cpp使用过程笔记**

- [安装及使用](#安装及使用--Linux系统)
  - [安装](#安装)
  - [量化](#量化)
  - [启动](#启动)
- [部署测试](#部署测试)
  - [InternLM2](#InternLM2)
  - [LLaMa3](#LLaMa3)
- [](#)
- [](#)
- [](#)

# 安装及使用--Linux系统

## 安装
 - 拉取llama.cpp仓库：git clone https://github.com/ggerganov/llama.cpp.git
 - cd llama.cpp
 - 编译
   - cpu: make
   - cuda: make LLAMA_CUDA=1

## 量化
 - 将Huggingface格式的模型下载到llama.cpp/models路径中
 - 针对不同模型，出权重文件外，可能还需要一些文件放在llama.cpp/models路径中
 - 转为16bit的gguf模型：
   - cd llama.cpp
   - python convert.py ./models/模型目录
     - 如果此步报错，尝试修改./models/模型目录/params.json，
将最后"vocab_size":中的值改为32000
 - 量化：
  - 4bit: **./quantize ./models/模型目录/ggml-model-f16.gguf ./models/模型目录/ggml-model-Q4_K_M.gguf -Q4_K_M**
  - 8bit: **./quantize ./models/模型目录/ggml-model-f16.gguf ./models/模型目录/ggml-model-Q8_K_M.gguf -Q8_K_M**


## 启动
 - GPU加速，只需在命令后加上参数"-ngl 1"，此参数可修改，最大为35
 - ./main -m ./models/mymodel/ggml-model-Q4_K_M.gguf -n 128 -t 18 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/chat-with-bob.txt -ngl 20
 - 上述命令中设置了部分参数，还有很多参数可以设置，主要参考此文档：https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md


# 部署测试

## InternLM2
&emsp;&emsp;最新版本(20240418)的llama.cpp应该是支持InternLM2，需要先将InternLM2模型转换为gguf，可按以下步骤部署
 - 本地下载InterLM2的HF模型，如internlm2-chat-7b
 - 进入llama.cpp的根目录: **cd your_path/llama.cpp**
 - 使用"convert-hf-to-gguf.py"脚本将internlm2-chat-7b转换为"ggml-model-f16.gguf": **python convert-hf-to-gguf.py your_path/internlm2-chat-7b**
 - 执行以下命令部署服务: **./main -m ./moedls/internlm2-chat-7b/ggml-model-f16.gguf --temp 0.2 --top-p 0.9 --top-k 5 --repeat_penalty 1.1 -ngl 10 --color -ins**

## LLaMa3
待定