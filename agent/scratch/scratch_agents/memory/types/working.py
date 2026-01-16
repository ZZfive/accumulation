"""工作记忆实现

- 短期上下文管理
- 容量和时间限制
- 优先级管理
- 自动清理机制
"""

import heapq
from typing import List, Dict, Any
from datetime import datetime, timedelta

from ..base import BaseMemory, MemoryItem, MemoryConfig