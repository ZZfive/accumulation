"""Function Call Agent - 基于OpenAI函数调用范式的Agent实现"""

from __future__ import annotations

import json
from typing import Iterator, Union, Dict, Any, TYPE_CHECKING, List

from ..core.llm import LLM
from ..core.agent import Agent
from ..core.config import Config
from ..core.message import Message
from ..core.exceptions import ScratchAgentsException

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry
    from ..tools.base import Tool


def _map_parameter_type(param_type: str) -> str:
    r"""将工具参数类型映射为JSON Schema允许的类型"""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


class FunctionCallAgent(Agent):
    r"""基于OpenAI原生函数调用机制的Agent"""

    def __init__(self, name: str, llm: LLM, system_prompt: str = None, config: Config = None,
                 tool_registry: "ToolRegistry" = None, enable_tool_calling: bool = True,
                 default_tool_choice: Union[str, Dict] = "auto", max_tool_iterations: int = 3) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations
    
    def _get_system_prompt(self) -> str:
        r"""构建系统提示词，注入工具描述"""
        base_prompt = self.system_prompt or "你是一个可靠的AI助理，能够在需要时调用工具完成任务。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n请主动决定是否调用工具，合理利用多次调用来获得完备答案。"
        return prompt
    
    def _build_tool_schema(self) -> List[Dict[str, Any]]:
        if not self.enable_tool_calling or not self.tool_registry:
            return []
        
        schemas: List[Dict[str, Any]] = []

        # Tool对象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: List[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []
            
            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)
        
        # register_function 注册的工具（直接访问内部结构）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas
    
    @staticmethod
    def _extract_messages_content(raw_content: Any) -> str:
        r"""从OpenAI响应的message.content中安全提取文本"""
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: List[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return str(raw_content)
    
    @staticmethod
    def _parse_function_call_arguments(arguments: str) -> Dict[str, Any]:
        r"""解析模型返回的JSON字符串参数"""
        if not arguments:
            return {}
        
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    
    def _convert_parameter_types(self, tool_name: str, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        r"""根据工具定义尽可能转换参数类型"""
        if not self.tool_registry:
            return param_dict
        
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict
        
        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict
        
        type_mapping = {param.name: param.type for param in tool_params}
        converted: Dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue
            
            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted
    
    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        r"""执行工具调用并返回字符串结果"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"
        
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as e:
                return f"❌ 错误：执行工具 '{tool_name}' 时发生异常: {str(e)}"
        
        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as e:
                return f"❌ 错误：执行工具 '{tool_name}' 时发生异常: {str(e)}"
        
        return f"❌ 错误：未找到名为 '{tool_name}' 的工具。"
    
    def _invoke_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                           tool_choice: Union[str, Dict] = "auto", **kwargs) -> str:
        r"""调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise ScratchAgentsException("LLM客户端未初始化，请检查LLM配置")
        
        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)
        
        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs
        )
    
    def run(self, input_text: str, *, max_tool_iterations: int = None,
            tool_choice: Union[str, Dict] = None, **kwargs) -> str:
        r"""执行函数调用范式的对话流程"""
        messages: List[Dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schema()
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(content=input_text, role="user"))
            self.add_message(Message(content=response_text, role="assistant"))
            return response_text
        
        iterations_limit = max_tool_iterations or self.max_tool_iterations
        effective_tool_choice: Union[str, Dict] = tool_choice or self.default_tool_choice

        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            response = self._invoke_with_tools(messages, tool_schemas, effective_tool_choice, **kwargs)

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_messages_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])

            if tool_calls:
                assistant_payload: Dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                messages.append(assistant_payload)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    messages.append({
                        "role": "tool",
                        "id": tool_call.id,
                        "name": tool_name,
                        "content": result
                    })
                current_iteration += 1
                continue
            
            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            final_choice = self._invoke_with_tools(messages, tool_schemas, "none", **kwargs)
            final_response = self._extract_messages_content(final_choice.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})
        
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_response, role="assistant"))
        return final_response
    
    def add_tool(self, tool: "Tool") -> None:
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ 工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return
        
        self.tool_registry.register_tool(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister_tool(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False
    
    def list_tools(self) -> List[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
    
    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式调用暂未实现，直接回退到一次性调用"""
        result = self.run(input_text, **kwargs)
        yield result