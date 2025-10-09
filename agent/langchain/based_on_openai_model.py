import os
from typing import Optional

from dotenv import load_dotenv

from pydantic import Field, SecretStr, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env

load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",  # 设置别名，允许通过api_key属性访问openai_api_key
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),  # 默认值从环境变量中获取
    )  # 用于从实例化的对象中获取密钥，因为使用了SecretStr，输出时会显示**********

    @property
    def lc_secrets(self) -> dict[str, str]:  # 密钥映射，用于LangChain框架的密钥管理
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):  # 初始化方法，用于实例化对象
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1",
                         openai_api_key=openai_api_key, **kwargs)


class ChatINTERNLM(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("INTERNLM_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "INTERNLM_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        openai_api_key = openai_api_key or os.environ.get("INTERNLM_API_KEY")
        super().__init__(base_url="https://chat.intern-ai.org.cn/api/v1",
                         openai_api_key=openai_api_key, **kwargs)


if __name__ == "__main__":
    # model = ChatOpenRouter(model="openrouter/sonoma-dusk-alpha")
    # response = model.invoke("Hello, how are you?")
    # print(response)

    openrouter_model = ChatINTERNLM(model="intern-latest")
    response = openrouter_model.invoke("Hello, how are you?")
    print(response)