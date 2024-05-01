**本文档用于记录vllm使用过程笔记**

- [安装](#安装)
- [](#)
- [](#)
- [性质](#性质)


# 安装
&emsp;&emsp;vllm安装步骤较为简单，在linux环境下，和常规使用的torch2.1.0+cu121结合，直接pip install vllm安装即可。其他安装环境参考官方文档：https://docs.vllm.ai/en/latest/getting_started/installation.html

# 性质
 - vllm兼容了huggingface和modelscope，按huggingface格式使用线上模型时，默认从huggingface下载模型，可以设置**export VLLM_USE_MODELSCOPE=True**环境参数从modelscope下载