"""
自定义异常类
"""

class ScratchAgentsException(Exception):
    """
    基础异常类
    """
    pass


class LLMException(ScratchAgentsException):
    """LLM相关异常"""
    pass


class AgentException(ScratchAgentsException):
    """Agent相关异常"""
    pass


class ConfigException(ScratchAgentsException):
    """配置相关异常"""
    pass


class ToolException(ScratchAgentsException):
    """工具相关异常"""
    pass