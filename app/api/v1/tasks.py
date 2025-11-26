"""任务查询API路由 - DashScope兼容的任务查询接口"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.auth import auth_manager
from app.core.logger import logger
from app.services.grok.task_manager import task_manager, TaskStatus


router = APIRouter(prefix="/v1", tags=["任务查询"])

# Bearer安全方案
tasks_security = HTTPBearer(auto_error=False)


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(tasks_security)
):
    """查询任务状态
    
    根据DashScope API文档，任务查询URL为 /v1/tasks/{task_id}
    """
    try:
        # 验证认证
        _ = auth_manager.verify(credentials)
        
        # 获取任务
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 构建响应
        # 兼容 DashScope 格式：将 SUCCESS 转换为 SUCCEEDED
        task_status = task["status"]
        if task_status == TaskStatus.SUCCESS.value:
            task_status = "SUCCEEDED"  # DashScope 使用 SUCCEEDED
        elif task_status == TaskStatus.FAILED.value:
            task_status = "FAILED"  # 保持 FAILED
        
        response = {
            "output": {
                "task_id": task_id,
                "task_status": task_status
            },
            "request_id": task_id
        }
        
        # 如果任务完成，添加结果
        if task["status"] == TaskStatus.SUCCESS.value:
            result = task.get("result", {})
            # 根据任务类型返回不同的结果格式
            task_type = task.get("task_type", "")
            if task_type in ["text2image", "image2image"]:
                # 图片任务返回图片URL列表
                # 兼容 DashScope 格式：使用 {url: "..."} 而不是 {image: "..."}
                image_urls = result.get("image_urls", [])
                if image_urls:
                    response["output"]["results"] = [{
                        "url": url  # DashScope 格式使用 "url" 字段
                    } for url in image_urls]
            elif task_type in ["text2video", "image2video"]:
                # 视频任务返回视频URL列表
                # 兼容 DashScope 格式：使用 {url: "..."} 而不是 {video: "..."}
                video_urls = result.get("video_urls", [])
                if video_urls:
                    response["output"]["results"] = [{
                        "url": url  # DashScope 格式使用 "url" 字段
                    } for url in video_urls]
        elif task["status"] == TaskStatus.FAILED.value:
            response["output"]["error"] = task.get("error", "未知错误")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Tasks] 查询任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

