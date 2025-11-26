"""任务管理器 - 用于跟踪异步任务状态"""

import uuid
import time
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

from app.core.logger import logger


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "PENDING"  # 等待中
    RUNNING = "RUNNING"  # 运行中
    SUCCESS = "SUCCESS"  # 成功
    FAILED = "FAILED"  # 失败


class TaskManager:
    """任务管理器（单例）"""
    
    _instance: Optional['TaskManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'TaskManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        logger.debug("[TaskManager] 初始化完成")
    
    def create_task(self, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建新任务
        
        Args:
            task_type: 任务类型（text2image, image2image, text2video, image2video）
            metadata: 任务元数据
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": TaskStatus.PENDING.value,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "metadata": metadata or {},
            "result": None,
            "error": None
        }
        logger.info(f"[TaskManager] 创建任务: {task_id} ({task_type})")
        return task_id
    
    def update_task(self, task_id: str, status: Optional[TaskStatus] = None, 
                   result: Optional[Any] = None, error: Optional[str] = None):
        """更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            result: 任务结果
            error: 错误信息
        """
        if task_id not in self._tasks:
            logger.warning(f"[TaskManager] 任务不存在: {task_id}")
            return
        
        task = self._tasks[task_id]
        if status:
            task["status"] = status.value
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error
            task["status"] = TaskStatus.FAILED.value
        
        task["updated_at"] = int(time.time())
        logger.debug(f"[TaskManager] 更新任务: {task_id} -> {task['status']}")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息字典，如果不存在返回None
        """
        return self._tasks.get(task_id)
    
    def delete_task(self, task_id: str):
        """删除任务
        
        Args:
            task_id: 任务ID
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.debug(f"[TaskManager] 删除任务: {task_id}")
    
    def cleanup_old_tasks(self, max_age: int = 3600):
        """清理旧任务
        
        Args:
            max_age: 最大保留时间（秒），默认1小时
        """
        current_time = int(time.time())
        to_delete = []
        
        for task_id, task in self._tasks.items():
            age = current_time - task.get("updated_at", 0)
            if age > max_age:
                to_delete.append(task_id)
        
        for task_id in to_delete:
            self.delete_task(task_id)
        
        if to_delete:
            logger.info(f"[TaskManager] 清理了 {len(to_delete)} 个旧任务")


# 全局实例
task_manager = TaskManager()

