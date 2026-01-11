"""Reflection Agent实现 - 自我反思与迭代优化的智能体"""

from typing import List, Dict, Any

from ..core.llm import LLM
from ..core.agent import Agent
from ..core.config import Config
from ..core.message import Message


# 默认提示词模板
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class Memroy:
    r"""
    简单的短期记忆模块，用于存储智能体的行动与反思轨迹
    """
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
    
    def add_record(self, record_type: str, content: str) -> None:
        """向记忆中添加一条新记录"""
        self.records.append({
            "type": record_type,
            "content": content
        })
        print(f"✅ 记忆已更新，新增一条: {record_type}记录")
    
    def get_trajectory(self) -> str:
        r"""将所有记忆记录格式化为一个连续的字符串文本"""
        trajectory = ""
        for record in self.records:
            if record["type"] == "execution":
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()
    
    def get_last_execution(self) -> str:
        r"""获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record['content']
        return ""
    
    def clear(self) -> None:
        r"""清空所有记忆记录"""
        self.records.clear()


class ReflectionAgent(Agent):
    r"""
    Reflection Agent - 自我反思与迭代优化的智能体

    这个Agent能够，
    1. 执行初始任务
    2，对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意

    特别适合代码生成，文档写作、分析报告等需要迭代优化的任务
    支持多种专业领域的提示词模板，用户可以自定义或使用内置模板
    """

    def __init__(self, name: str, llm: LLM, system_prompt: str = None, config: Config = None,
                 max_iterations: int = 3, custom_prompts: Dict[str, str] = None) -> None:
        r"""
        初始化ReflectionAgent

        Args：
        name：Agent名称
        llm：LLM实例
        system_prompt：系统提示词
        config：配置对象
        max_iterations：最大迭代次数
        custom_prompts：自定义提示词字典
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memroy()

        self.prompts = custom_prompts or DEFAULT_PROMPTS
    
    def run(self, input_text: str, **kwargs) -> str:
        r"""
        运行Reflection Agent

        Args:
            input_text：用户输入的初始任务描述
            **kwargs：其他可选参数

        Returns:
            最终优化后的结果
        """
        print(f"\n🤖 {self.name} 开始执行任务: {input_text}")

        # 重置记忆
        self.memory.clear()

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        inital_prompt = self.prompts["initial"].format(task=input_text)
        intial_result = self._get_llm_response(inital_prompt, **kwargs)
        self.memory.add_record("execution", intial_result)

        # 2. 迭代循环，反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(task=input_text, content=last_result)
            feedback = self._get_llm_response(reflect_prompt, **kwargs)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("✅ 任务已完成，无需更多改进")
                break
            
            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(task=input_text, last_attempt=last_result, feedback=feedback)
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)
            
        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        # 保存结果
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_result, role="assistant"))

        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用LLM并获取完整响应"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""