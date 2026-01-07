"""简单Agent实现-基于OpenAI原生API"""

import re
from typing import Optional, Iterable, TYPE_CHECKING

from ..core.llm import LLM
from ..core.agent import Agent
from ..core.config import Config
from ..core.message import Message

