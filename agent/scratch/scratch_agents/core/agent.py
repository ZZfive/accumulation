"""Agent基类"""

from typing import List
from abc import ABC, abstractmethod

from .llm import LLM
from .config import Config
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(self, name: str, llm: LLM, system_prompt: str = None, config: Config = None) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息"""
        self._history.append(message)
    
    def get_history(self) -> List[Message]:
        """获取历史消息"""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """清空历史消息"""
        self._history.clear()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    def __repr__(self) -> str:
        return self.__str__()