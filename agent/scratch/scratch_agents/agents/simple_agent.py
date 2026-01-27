"""简单Agent实现-基于OpenAI原生API"""

import re
import json
from typing import Iterator, TYPE_CHECKING, List, Dict

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

    def _parse_tool_calls(self, text: str) -> List:
        r"""解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)  # 返回所有匹配的工具调用

        tool_calls = []
        for tool_name, params in matches:
            tool_calls.append({
                "tool_name": tool_name.strip(),
                "parameters": params.strip(),
                "original_text": f"[TOOL_CALL:{tool_name}:{params}]"
            })

        return tool_calls
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        r"""执行工具调用"""
        if not self.tool_registry:
            return f"错误：工具注册表未配置，无法执行工具调用。"
        
        try:
            # 获取Tool对象
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"错误：未找到名为 '{tool_name}' 的工具。"
            
            # 解析参数
            params_dict = self._parse_tool_parameters(parameters)
            
            # 执行工具
            result = tool.run(params_dict)
            return f"工具 '{tool_name}' 执行结果：{result}"
        except Exception as e:
            return f"错误：执行工具 '{tool_name}' 时发生异常: {str(e)}"
    
    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> Dict:
        r"""智能解析工具参数"""
        params_dict = {}

        if parameters.strip().startswith('{'):
            try:
                params_dict = json.loads(parameters)
                params_dict = self._convert_params_types(tool_name, params_dict)
                return params_dict
            except json.JSONDecodeError:
                pass
        
        if "=" in parameters:
            # 格式 key=value 或action=search，query=Python
            if ":" in parameters:
                # 多个参数
                pairs = parameters.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        params_dict[key.strip()] = value.strip()
            else:
                # 单个参数
                key, value = parameters.split("=", 1)
                params_dict[key.strip()] = value.strip()

            # 类型转换
            params_dict = self._convert_params_types(tool_name, params_dict)

            # 推断action
            if "action" not in params_dict:
                params_dict = self._infer_action(tool_name, params_dict)
        else:
            # 直接传入参数，根据工具类型智能推断
            params_dict = self._infer_simple_parameters(tool_name, parameters)
        
        return params_dict
    
    def _convert_params_types(self, tool_name: str, params_dict: Dict) -> Dict:
        r"""
        根据工具的参数定义转换参数类型

        Args:
            tool_name: 工具名称
            params_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return params_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return params_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        except:
            return params_dict

        # 创建参数类型映射
        params_types = {}
        for param in tool_params:
            params_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in params_dict.items():
            if key in params_types:
                param_type = params_types[key]
                try:
                    if param_type == 'number' or param_type == 'integer':
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, params_dict: Dict) -> Dict:
        """根据工具类型和参数推断action"""
        if tool_name == 'memory':
            if 'recall' in params_dict:
                params_dict['action'] = 'search'
                params_dict['query'] = params_dict.pop('recall')
            elif 'store' in params_dict:
                params_dict['action'] = 'add'
                params_dict['content'] = params_dict.pop('store')
            elif 'query' in params_dict:
                params_dict['action'] = 'search'
            elif 'content' in params_dict:
                params_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in params_dict:
                params_dict['action'] = 'search'
                params_dict['query'] = params_dict.pop('search')
            elif 'query' in params_dict:
                params_dict['action'] = 'search'
            elif 'text' in params_dict:
                params_dict['action'] = 'add_text'

        return params_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        运行SimpleAgent，支持可选的工具调用
        
        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            **kwargs: 其他参数
            
        Returns:
            Agent响应
        """
        # 构建消息列表
        messages = []
        
        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})
        
        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})
        
        # 如果没有启用工具调用，使用原有逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(content=input_text, role="user"))
            self.add_message(Message(content=response, role="assistant"))
            return response
        
        # 迭代处理，支持多轮工具调用
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 调用LLM
            response = self.llm.invoke(messages, **kwargs)  # LLM会基于系统提示词、历史消息和当前用户消息生成响应，自动判断是否需要使用工具

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original_text'], "")

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)
        
        # 保存到历史记录
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_response, role="assistant"))

        return final_response

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool对象
            auto_expand: 是否自动展开可展开的工具（默认True）

        如果工具是可展开的（expandable=True），会自动展开为多个独立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 会自动处理工具展开
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent
        
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
            
        Yields:
            Agent响应片段
        """
        # 构建消息列表
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # 流式调用LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # 保存完整对话到历史记录
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=full_response, role="assistant"))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    llm = LLM()

    # 测试1:基础对话Agent（无工具）
    print("=== 测试1:基础对话 ===")
    basic_agent = SimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    response1 = basic_agent.run("你好，请介绍一下自己")
    print(f"基础对话响应: {response1}\n")

    # 测试2:带工具的Agent
    from scratch_agents.tools.builtin import CalculatorTool
    from scratch_agents.tools.registry import ToolRegistry

    print("=== 测试2:工具增强对话 ===")
    tool_registry = ToolRegistry()
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)

    enhanced_agent = SimpleAgent(
        name="增强助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=tool_registry,
        enable_tool_calling=True
    )

    response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32")
    print(f"工具增强响应: {response2}\n")

    # 测试3:流式响应
    print("=== 测试3:流式响应 ===")
    print("流式响应: ", end="")
    for chunk in basic_agent.stream_run("请解释什么是人工智能"):
        pass  # 内容已在stream_run中实时打印

    # 测试4:动态添加工具
    print("\n=== 测试4:动态工具管理 ===")
    print(f"添加工具前: {basic_agent.has_tools()}")
    basic_agent.add_tool(calculator)
    print(f"添加工具后: {basic_agent.has_tools()}")
    print(f"可用工具: {basic_agent.list_tools()}")

    # 查看对话历史
    print(f"\n对话历史: {len(basic_agent.get_history())} 条消息")