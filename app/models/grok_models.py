"""Grok 模型配置和枚举定义"""

import re
from enum import Enum
from typing import Dict, Any, Tuple


# 动态 modelMode 前缀
_DYNAMIC_MODE_PREFIX = "MODEL_MODE_"

# modelName 验证正则：必须以字母或数字开头，只能包含字母、数字、连字符、下划线、点
_MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$')


def _is_dynamic_model(model: str) -> bool:
    """检查是否为动态 modelName-modelMode 格式
    
    格式: modelName-MODEL_MODE_XXX
    例如: boj-MODEL_MODE_FAST, custom-model-MODEL_MODE_HEAVY
    """
    if "-" not in model:
        return False
    
    # 从右侧分割，获取最后一段作为 mode
    parts = model.rsplit("-", 1)
    if len(parts) != 2:
        return False
    
    model_name, mode = parts
    
    # 验证 modelName
    if not model_name or not _MODEL_NAME_PATTERN.match(model_name):
        return False
    
    # 验证 mode 格式（必须以 MODEL_MODE_ 开头）
    if not mode.startswith(_DYNAMIC_MODE_PREFIX):
        return False
    
    return True


def _parse_dynamic_model(model: str) -> Tuple[str, str]:
    """解析动态模型，返回 (modelName, modelMode)
    
    前提: 已通过 _is_dynamic_model 验证
    """
    model_name, mode = model.rsplit("-", 1)
    return model_name, f"{_DYNAMIC_MODE_PREFIX}{mode[len(_DYNAMIC_MODE_PREFIX):]}"




# 模型配置
_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "grok-3-fast": {
        "grok_model": ("grok-3", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 3 Fast",
        "description": "Fast and efficient Grok 3 model",
        "raw_model_path": "xai/grok-3",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-fast": {
        "grok_model": ("grok-4-mini-thinking-tahoe", "MODEL_MODE_GROK_4_MINI_THINKING"),
        "rate_limit_model": "grok-4-mini-thinking-tahoe",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Fast",
        "description": "Fast version of Grok 4 with mini thinking capabilities",
        "raw_model_path": "xai/grok-4-mini-thinking-tahoe",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-fast-expert": {
        "grok_model": ("grok-4-mini-thinking-tahoe", "MODEL_MODE_EXPERT"),
        "rate_limit_model": "grok-4-mini-thinking-tahoe",
        "cost": {"type": "high_cost", "multiplier": 4, "description": "计4次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Fast Expert",
        "description": "Expert mode of Grok 4 Fast with enhanced reasoning",
        "raw_model_path": "xai/grok-4-mini-thinking-tahoe",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-expert": {
        "grok_model": ("grok-4", "MODEL_MODE_EXPERT"),
        "rate_limit_model": "grok-4",
        "cost": {"type": "high_cost", "multiplier": 4, "description": "计4次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Expert",
        "description": "Full Grok 4 model with expert mode capabilities",
        "raw_model_path": "xai/grok-4",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-heavy": {
        "grok_model": ("grok-4-heavy", "MODEL_MODE_HEAVY"),
        "rate_limit_model": "grok-4-heavy",
        "cost": {"type": "independent", "multiplier": 1, "description": "独立计费，只有Super用户可用"},
        "requires_super": True,
        "display_name": "Grok 4 Heavy",
        "description": "Most powerful Grok 4 model with heavy computational capabilities. Requires Super Token for access.",
        "raw_model_path": "xai/grok-4-heavy",
        "default_temperature": 1.0,
        "default_max_output_tokens": 65536,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1": {
        "grok_model": ("grok-4-1-non-thinking-w-tool", "MODEL_MODE_GROK_4_1"),
        "rate_limit_model": "grok-4-1-non-thinking-w-tool",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1",
        "description": "Latest Grok 4.1 model with tool capabilities",
        "raw_model_path": "xai/grok-4-1-non-thinking-w-tool",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1-thinking": {
        "grok_model": ("grok-4-1-thinking-1108b", "MODEL_MODE_AUTO"),
        "rate_limit_model": "grok-4-1-thinking-1108b",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Thinking",
        "description": "Grok 4.1 model with advanced thinking and tool capabilities",
        "raw_model_path": "xai/grok-4-1-thinking-1108b",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1-thinking-1129": {
        "grok_model": ("grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_THINKING"),
        "rate_limit_model": "grok-4-1-thinking-1129",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Thinking 1129",
        "description": "Grok 4.1 Thinking model (1129 version) with enhanced reasoning",
        "raw_model_path": "xai/grok-4-1-thinking-1129",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-imagine-0.9": {
        "grok_model": ("grok-3", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok Imagine 0.9",
        "description": "Image and video generation model. Supports text-to-image and image-to-video generation.",
        "raw_model_path": "xai/grok-imagine-0.9",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95,
        "is_video_model": True
    }
}


class TokenType(Enum):
    """Token类型"""
    NORMAL = "ssoNormal"
    SUPER = "ssoSuper"


class Models(Enum):
    """支持的模型"""
    GROK_3_FAST = "grok-3-fast"
    GROK_4_1 = "grok-4.1"
    GROK_4_1_THINKING = "grok-4.1-thinking"
    GROK_4_1_THINKING_1129 = "grok-4.1-thinking-1129"
    GROK_4_FAST = "grok-4-fast"
    GROK_4_FAST_EXPERT = "grok-4-fast-expert"
    GROK_4_EXPERT = "grok-4-expert"
    GROK_4_HEAVY = "grok-4-heavy"
    GROK_IMAGINE_0_9 = "grok-imagine-0.9"

    @classmethod
    def get_model_info(cls, model: str) -> Dict[str, Any]:
        """获取模型配置"""
        return _MODEL_CONFIG.get(model, {})

    @classmethod
    def is_valid_model(cls, model: str) -> bool:
        """检查模型是否有效
        
        支持:
        1. 预定义模型（如 grok-4-heavy）
        2. 动态 modelName-modelMode 格式（如 boj-MODEL_MODE_FAST）
        """
        # 预定义模型
        if model in _MODEL_CONFIG:
            return True
        
        # 动态 modelName-modelMode 格式
        return _is_dynamic_model(model)

    @classmethod
    def to_grok(cls, model: str) -> Tuple[str, str]:
        """转换为Grok内部模型名和模式

        Returns:
        (模型名, 模式类型) 元组
        
        支持:
        1. 预定义模型 - 使用配置中的映射
        2. 动态 modelName-modelMode 格式 - 解析为 (modelName, modelMode)
        3. 其他 - 默认返回 (model, MODEL_MODE_FAST)
        """
        config = _MODEL_CONFIG.get(model)
        if config:
            return config["grok_model"]
        
        # 动态解析 modelName-modelMode
        if _is_dynamic_model(model):
            model_name, mode = _parse_dynamic_model(model)
            return (model_name, mode)
        
        # 默认回退
        return (model, "MODEL_MODE_FAST")

    @classmethod
    def to_rate_limit(cls, model: str) -> str:
        """转换为速率限制模型名
        
        支持:
        1. 预定义模型 - 使用配置中的 rate_limit_model
        2. 动态 modelName-modelMode 格式 - 使用 modelName 作为速率限制标识
        3. 其他 - 原样返回
        """
        config = _MODEL_CONFIG.get(model)
        if config:
            return config["rate_limit_model"]
        
        # 动态模型使用 modelName 部分
        if _is_dynamic_model(model):
            model_name, _ = _parse_dynamic_model(model)
            return model_name
        
        return model

    @classmethod
    def get_all_model_names(cls) -> list[str]:
        """获取所有模型名称"""
        return list(_MODEL_CONFIG.keys())
