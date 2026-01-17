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

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class WorkingMemory(BaseMemory):
    r"""工作记忆实现
    
    特点:
    - 优先容量
    - 时效性强，会话级别
    - 优先级管理
    - 自动清理过期记忆
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 工作记忆特定配置
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        # 纯内存TTL，可通过MemoryConfig上挂载working_memory_ttl_minutes覆盖
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        # 内存存储，工作记忆不需要持久化
        self.memories: List[MemoryItem] = []

        # 使用优先队列管理记忆
        self.memory_queue = []  # (priority, timestamp, memory_item)

    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项"""
        # 过期清理
        self._expire_old_memories()
        # 计算优先级（重要性 + 时间衰减）
        priority = self._calculate_priority(memory_item)

        # 添加到堆中
        heapq.heappush(self.memory_queue, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)

        # 更新当前记忆量
        self.current_tokens += len(memory_item.content.split())

        # 检查容量限制
        self._enforce_capacity_limits()

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:
        r"""检索工作记忆 - 混合记忆向量检索和关键词匹配"""
        # 过期清理
        self._expire_old_memories()
        if not self.memories:
            return []
        
        # 过滤已遗忘的记忆
        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]

        # 按用户ID过滤
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in filtered_memories if m.user_id == user_id]
        
        if not filtered_memories:
            return []
        
        # 尝试语义向量检索
        vector_scores = {}
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError

            # 准备文档
            documents = [query] + [m.content for m in filtered_memories]

            # TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)

            # 计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # 存储向量分数
            for i, m in enumerate(filtered_memories):
                vector_scores[m.id] = similarities[i]
        except Exception as e:
            # 语义检索失败，回退到关键词匹配
            vector_scores = {}
        
        # 计算最终分数
        query_lower = query.lower()
        scored_memories = []

        for m in filtered_memories:
            content_lower = m.content.lower()

            # 获取向量分数
            vector_score = vector_scores.get(m.id, 0.0)

            # 关键词匹配
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # 分词匹配
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8
            
            # 混合分数：向量检索 + 关键词匹配
            if vector_score > 0.0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score
            
            # 时间衰减
            time_decay = self._calculate_time_decay(m.timestamp)
            base_relevance *= time_decay

            # 重要性权重
            importance_weight = 0.8 + (m.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0.0:
                scored_memories.append((final_score, m))
        
        # 排序并返回
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:limit]]

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        r"""更新工作记忆"""
        for m in self.memories:
            if m.id == memory_id:
                old_tokens = len(m.content.split())

                if content is not None:
                    m.content = content
                    # 更新token计数
                    new_tokens = len(content.split())
                    self.current_tokens += new_tokens - old_tokens
                
                if importance is not None:
                    m.importance = importance
                
                if metadata is not None:
                    m.metadata.update(metadata)

                # 重新计算优先级
                self._update_heap_priority(m)
                
                return True
        
        return False

    def remove(self, memory_id: str) -> bool:
        r"""删除工作记忆"""
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                # 从列表中删除
                removed_memory = self.memories.pop(i)

                # 从堆中删除
                self._mark_deleted_in_heap(memory_id)

                # 更新token计数
                self.current_tokens -= len(removed_memory.content.split())
                self.current_tokens = max(0, self.current_tokens)

                return True
        return False
    
    def has_memory(self, memory_id: str) -> bool:
        r"""检查记忆是否存在"""
        for m in self.memories:
            if m.id == memory_id:
                return True
        return False
    
    def clear(self):
        r"""清空工作记忆"""
        self.memories.clear()
        self.memory_queue.clear()
        self.current_tokens = 0
        # self.seesion_start = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        r"""获取工作记忆统计信息"""
        # 过期清理（惰性）
        self._expire_old_memories()

        # 工作记忆中的记忆都是活跃的（已遗忘的记忆会被直接删除）
        active_memories = self.memories

        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 工作记忆中已遗忘的记忆会被直接删除
            "total_count": len(self.memories),  # 总记忆数量
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }
    
    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        r"""获取最近记忆"""
        sorted_memories = sorted(self.memories, key=lambda x: x.timestamp, reverse=True)
        return sorted_memories[:limit]
    
    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        r"""获取重要记忆"""
        sorted_memories = sorted(self.memories, key=lambda x: x.importance, reverse=True)
        return sorted_memories[:limit]
    
    def get_all(self) -> List[MemoryItem]:
        r"""获取所有记忆"""
        return self.memories.copy()
    
    def get_context_summary(self, max_length: int = 500) -> str:
        r"""获取上下文摘要"""
        if not self.memories:
            return "No working memories available"
        
        # 按重要性和时间排序
        sorted_memories = sorted(self.memories, key=lambda x: (x.importance, x.timestamp), reverse=True)

        summary_parts = []
        current_length = 0

        for m in sorted_memories:
            content = m.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                # 截取最后一个记忆
                remaining = max_length - current_length
                if remaining > 50:
                    summary_parts.append(content[:remaining] + "...")
                break
        
        return "Working Memory Context:\n" + "\n".join(summary_parts)
    
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 1) -> int:
        r"""工作记忆遗忘机制"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove = []
        
        # 始终先执行TTL过期（分钟级）
        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)
        
        if strategy == "importance_based":
            # 删除低重要性记忆
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        
        elif strategy == "time_based":
            # 删除过期记忆（工作记忆通常以小时计算）
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)
        
        elif strategy == "capacity_based":
            # 删除超出容量的记忆
            if len(self.memories) > self.max_capacity:
                # 按优先级排序，删除最低的
                sorted_memories = sorted(
                    self.memories,
                    key=lambda m: self._calculate_priority(m)
                )
                excess_count = len(self.memories) - self.max_capacity
                for memory in sorted_memories[:excess_count]:
                    to_remove.append(memory.id)
        
        # 执行删除
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
        
        return forgotten_count

    def _calculate_priority(self, memory_item: MemoryItem) -> float:
        r"""计算记忆优先级：重要性 + 时间衰减"""
        priority = memory_item.importance

        # 时间衰减（越久越低）
        time_decay = self._calculate_time_decay(memory_item.timestamp)
        priority *= time_decay

        return priority
    
    def _calculate_time_decay(self, timestamp: datetime) -> float:
        r"""计算时间衰减因子"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600

        # 指数衰减，工作记忆衰减更快
        decay_factor = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay_factor)
    
    def _enforce_capacity_limits(self):
        r"""强制执行容量限制"""
        # 检查记忆数量限制
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        
        # 检查token数量限制
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self):
        r"""按TTL清理过期记忆，并同步更新堆与token计数"""
        if not self.memories:
            return
        
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        # 过滤保留的记忆
        kept = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp >= cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())
        if len(kept) == len(self.memories):
            return
        
        # 覆盖列表与token
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)

        # 重建堆
        self.memory_queue = []
        for m in self.memories:
            priority = self._calculate_priority(m)
            heapq.heappush(self.memory_queue, (-priority, m.timestamp, m))
    
    def _remove_lowest_priority_memory(self):
        r"""移除优先级最低的记忆"""
        if not self.memory_queue:
            return
        
        # 找到优先级最低的记忆
        lowest_priority = float('inf')
        lowest_memory = None

        for m in self.memories:
            priority = self._calculate_priority(m)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = m
        
        if lowest_memory:
            self.remove(lowest_memory.id)
    
    def _update_heap_priority(self, memory_item: MemoryItem):
        r"""更新堆中记忆的优先级"""
        self.memory_queue = []
        for m in self.memories:
            priority = self._calculate_priority(m)
            heapq.heappush(self.memory_queue, (-priority, m.timestamp, m))
    
    def _mark_deleted_in_heap(self, memory_id: str):
        """在堆中标记删除的记忆"""
        # 由于heapq不支持直接删除，先标记为已删除，然后在后续的操作中会被清理
        pass