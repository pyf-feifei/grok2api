#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LiteLLM 服务启动入口 - 类似 main.py 的方式"""

import os
import sys
from pathlib import Path

# 设置环境变量确保 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 设置标准输出编码为 UTF-8（Windows）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保配置文件存在
config_file = Path("litellm_config.yaml")
if not config_file.exists():
    print("[INFO] 配置文件不存在，正在生成...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "create_litellm_config.py"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            print("[OK] 配置文件已生成")
        else:
            print(f"[ERROR] 生成配置文件失败: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 无法生成配置文件: {e}")
        sys.exit(1)

# 读取配置以显示信息
try:
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    server_settings = config.get('server_settings', {})
    port = server_settings.get('port', 4000)
    host = server_settings.get('host', '0.0.0.0')
    
    model_list = config.get('model_list', [])
    print(f"[LiteLLM] 启动服务...")
    print(f"  配置文件: {config_file}")
    print(f"  监听地址: {host}:{port}")
    print(f"  模型数量: {len(model_list)}")
    if model_list:
        model_names = [m.get('model_name', 'unknown') for m in model_list[:3]]
        print(f"  模型列表: {', '.join(model_names)}{'...' if len(model_list) > 3 else ''}")
except Exception as e:
    print(f"[WARNING] 无法读取配置信息: {e}")
    port = 4000
    host = '0.0.0.0'

print("\n" + "="*50)
print("LiteLLM 服务启动中...")
print("按 Ctrl+C 停止服务")
print("="*50 + "\n")

# 启动 LiteLLM 服务
if __name__ == "__main__":
    try:
        import subprocess
        import shutil
        
        # 查找 litellm 可执行文件
        # 1. 优先在当前 Python 环境的 Scripts 目录查找 (venv 环境)
        scripts_dir = os.path.dirname(sys.executable)
        executable_name = "litellm.exe" if sys.platform == "win32" else "litellm"
        litellm_path = os.path.join(scripts_dir, executable_name)
        
        # 2. 如果当前环境没找到，尝试在 PATH 中查找
        if not os.path.exists(litellm_path):
            litellm_path = shutil.which("litellm")
            
        # 3. 如果还是找不到，尝试构建 uv run 命令 (如果用户安装了 uv)
        if not litellm_path and shutil.which("uv"):
            litellm_path = "uv"
            base_args = ["run", "litellm"]
        else:
            base_args = []

        if litellm_path:
            if base_args:
                cmd = [litellm_path] + base_args
            else:
                cmd = [litellm_path]
                
            cmd.extend([
                "--config",
                str(config_file),
                "--port",
                str(port),
                "--host",
                host
            ])
            
            print(f"[INFO] 启动命令: {' '.join(cmd)}")
            print()
            
            # 直接运行，实时显示日志
            subprocess.run(cmd, check=True)
            
        else:
            print("[ERROR] 找不到 litellm 可执行文件")
            print(f"  搜索路径: {os.path.join(scripts_dir, executable_name)}")
            print("  请确保已安装: pip install litellm")
            print("  或使用 uv: uv sync")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INFO] LiteLLM 服务已停止")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

