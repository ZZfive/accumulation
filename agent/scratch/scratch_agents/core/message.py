"""消息系统"""

from datetime import datetime
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)  # 使用 default_factory
    metadata: Dict[str, Any] = Field(default_factory=dict)     # 避免可变默认值问题
    
    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"