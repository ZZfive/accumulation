"""简单Agent实现-基于OpenAI原生API"""

import re
from typing import Optional, Iterable, TYPE_CHECKING

from ..core.llm import LLM
from ..core.agent import Agent
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class SimpleAgent(Agent):
    r"""简单的对话Agent，支持可选的工具调用"""

    def __init__(self, name: str, llm: LLM, system_prompt: str = None,
                 config: Config = None, tool_registry: "ToolRegistry" = None,
                 enable_tool_calling: bool = True) -> None:
                 r"""
                 初始化SimpleAgent

                 Args：
                    name：Agent名称
                    llm：LLM实例
                    system_prompt：系统提示词
                    confif：配置对象
                    tool_registry：工具注册表
                    enable_tool_calling：是否启用工具调用，只有在提供tool_registry时生效
                 """
                 super().__init__(name, llm, system_prompt, config)
                 self.tool_registry = tool_registry
                 self.enable_tool_calling = enable_tool_calling and tool_registry is not name
    
    def _get_enhanced_system_prompt(self) -> str:
        r"""构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式说明\n"
        tools_section += "1. **多个参数**：使用 `key=value` 格式，用逗号分隔\n"
        tools_section += "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **单个参数**：直接使用 `key=value`\n"
        tools_section += "   示例：`[TOOL_CALL:search:query=Python编程]`\n\n"
        tools_section += "3. **简单查询**：可以直接传入文本\n"
        tools_section += "   示例：`[TOOL_CALL:search:Python编程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section