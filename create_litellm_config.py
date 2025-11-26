#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""创建 LiteLLM 配置文件（UTF-8 编码）"""

import toml
from pathlib import Path

# 读取 Grok2API 配置
config_path = Path("data/setting.toml")
grok2api_url = "http://localhost:8001/v1"
api_key = "sk-any-key"

if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    # 读取 base_url
    global_config = config.get("global", {})
    base_url = global_config.get("base_url", "http://localhost:8001")
    if base_url:
        grok2api_url = base_url.rstrip("/")
        if not grok2api_url.endswith("/v1"):
            grok2api_url = grok2api_url + "/v1"

    # 读取 api_key
    grok_config = config.get("grok", {})
    api_key_value = grok_config.get("api_key", "")
    if api_key_value:
        api_key = api_key_value
    else:
        api_key = "sk-any-key"

# 生成配置文件内容
config_content = f"""# LiteLLM 配置文件 - 包含所有 Grok2API 支持的模型
# 此文件由 create_litellm_config.py 自动生成

model_list:
  - model_name: grok-4.1
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-4.1-thinking
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-4-fast
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-4-fast-expert
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-4-expert
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-4-heavy
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-3-fast
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120
  - model_name: grok-imagine-0.9
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 300
  - model_name: grok-model
    litellm_params:
      model: openai/custom
      api_base: {grok2api_url}
      api_key: {api_key}
      timeout: 120

general_settings:
  debug: true
  log_level: "INFO"

server_settings:
  host: "0.0.0.0"
  port: 4000
"""

# 保存为 UTF-8 无 BOM（明确指定 newline='\n' 确保跨平台兼容）
output_path = Path("litellm_config.yaml")
with open(output_path, "w", encoding="utf-8", newline='\n') as f:
    f.write(config_content)

# 验证文件编码
try:
    with open(output_path, "rb") as f:
        raw_data = f.read()
    # 检查是否是有效的 UTF-8
    raw_data.decode('utf-8')
except UnicodeDecodeError:
    print("[WARNING] 文件编码验证失败，重新生成...")
    # 重新写入，确保 UTF-8
    with open(output_path, "wb") as f:
        f.write(config_content.encode('utf-8'))

print(f"[OK] 配置文件已生成: {output_path}")
print(f"  Grok2API: {grok2api_url}")
print(f"  API Key: {api_key}")
