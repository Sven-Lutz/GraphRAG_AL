# crawler/core/scheduler.py
from __future__ import annotations

import heapq
from dataclasses import dataclass, field 
from typing import List, Optional

from crawler.core.models import CrawlTask

@dataclass(order=True, slots=True)
class _HeapItem:
    neg_priority: int
    depth: int
    seq: int
    task: CrawlTask = field(compare=False) 

class PriorityScheduler:

    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: List[_HeapItem] = []
        self._seq: int = 0

    def __len__(self) -> int:
        return len(self._heap)

    def enqueue(self, task: CrawlTask, priority: int) -> None:
        self._seq += 1
        heapq.heappush(
            self._heap,
            _HeapItem(
                neg_priority=-int(priority),
                depth=int(task.depth),
                seq=self._seq,
                task=task,
            ),
        )

    def has_next(self) -> bool:
        return bool(self._heap)

    def next(self) -> CrawlTask:
        if not self._heap:
            raise IndexError("PriorityScheduler.next() called on empty queue")
        return heapq.heappop(self._heap).task

    def peek(self) -> Optional[CrawlTask]:
        return self._heap[0].task if self._heap else None

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0