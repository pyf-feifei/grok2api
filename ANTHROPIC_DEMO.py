"""
Anthropic API 完整演示
演示如何使用 Anthropic SDK 与 Grok2API 交互
"""

import os
from typing import Optional

# 检查是否安装了 anthropic
try:
    from anthropic import Anthropic, Stream
    from anthropic.types import Message
except ImportError:
    print("=" * 70)
    print("❌ 错误: 未安装 anthropic SDK")
    print("=" * 70)
    print("\n请运行以下命令安装:")
    print("  pip install anthropic")
    print("\n或使用 uv:")
    print("  uv pip install anthropic")
    print()
    exit(1)


class Grok2APIAnthropicClient:
    """Grok2API Anthropic 客户端封装"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8002/v1"
    ):
        """
        初始化客户端
        
        Args:
            api_key: API 密钥（默认从环境变量 GROK2API_API_KEY 读取）
            base_url: 服务地址（默认本地 8002 端口）
        """
        self.api_key = api_key or os.getenv("GROK2API_API_KEY", "test-key")
        self.base_url = base_url
        
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✅ 客户端初始化成功")
        print(f"   - API Key: {self.api_key[:10]}...")
        print(f"   - Base URL: {self.base_url}")
        print()
    
    def chat(
        self,
        message: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: float = 1.0
    ) -> str:
        """
        发送单条消息
        
        Args:
            message: 用户消息
            model: 模型名称（Claude 模型名会自动映射到 Grok）
            max_tokens: 最大 token 数
            system: 系统提示词
            temperature: 温度参数
            
        Returns:
            助手回复的文本
        """
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": message}]
            }
            
            if system:
                kwargs["system"] = system
            
            response = self.client.messages.create(**kwargs)
            
            # 提取文本内容
            text = response.content[0].text if response.content else ""
            
            # 打印使用信息
            print(f"📊 Token 使用: {response.usage.input_tokens} 输入, "
                  f"{response.usage.output_tokens} 输出")
            
            return text
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            raise
    
    def chat_stream(
        self,
        message: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        system: Optional[str] = None
    ):
        """
        发送消息并流式接收响应
        
        Args:
            message: 用户消息
            model: 模型名称
            max_tokens: 最大 token 数
            system: 系统提示词
            
        Yields:
            响应的文本片段
        """
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}]
            }
            
            if system:
                kwargs["system"] = system
            
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            print(f"❌ 错误: {e}")
            raise


def demo_basic_chat():
    """演示 1: 基础对话"""
    print("=" * 70)
    print("演示 1: 基础对话")
    print("=" * 70)
    
    client = Grok2APIAnthropicClient()
    
    response = client.chat(
        message="你好！用中文简单介绍一下量子计算。",
        model="claude-3-5-sonnet-20241022",
        max_tokens=512
    )
    
    print(f"🤖 回复:\n{response}")
    print()


def demo_system_prompt():
    """演示 2: 使用系统提示词"""
    print("=" * 70)
    print("演示 2: 系统提示词")
    print("=" * 70)
    
    client = Grok2APIAnthropicClient()
    
    response = client.chat(
        message="如何读取 JSON 文件？",
        system="你是一个 Python 专家，总是用简洁的代码示例回答。",
        max_tokens=512
    )
    
    print(f"🤖 回复:\n{response}")
    print()


def demo_streaming():
    """演示 3: 流式响应"""
    print("=" * 70)
    print("演示 3: 流式响应")
    print("=" * 70)
    
    client = Grok2APIAnthropicClient()
    
    print("🤖 回复: ", end="", flush=True)
    
    for chunk in client.chat_stream(
        message="讲一个关于人工智能的小故事",
        max_tokens=512
    ):
        print(chunk, end="", flush=True)
    
    print("\n")


def demo_different_models():
    """演示 4: 不同的模型"""
    print("=" * 70)
    print("演示 4: 使用不同模型")
    print("=" * 70)
    
    client = Grok2APIAnthropicClient()
    
    models = [
        "claude-3-5-sonnet-20241022",  # 映射到 grok-2-latest
        "claude-3-haiku-20240307",     # 映射到 grok-2-1212
    ]
    
    for model in models:
        print(f"\n📌 使用模型: {model}")
        try:
            response = client.chat(
                message="说一个数字",
                model=model,
                max_tokens=50
            )
            print(f"🤖 回复: {response}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print()


def demo_temperature():
    """演示 5: 温度参数"""
    print("=" * 70)
    print("演示 5: 温度参数对比")
    print("=" * 70)
    
    client = Grok2APIAnthropicClient()
    
    for temp in [0.1, 1.0, 1.5]:
        print(f"\n📌 温度: {temp}")
        try:
            response = client.chat(
                message="用一个词形容人工智能",
                temperature=temp,
                max_tokens=20
            )
            print(f"🤖 回复: {response}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 Grok2API - Anthropic 接口演示")
    print("=" * 70)
    print()
    
    # 检查配置
    api_key = os.getenv("GROK2API_API_KEY", "test-key")
    base_url = os.getenv("GROK2API_BASE_URL", "http://localhost:8002/v1")
    
    print("📝 配置信息:")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Base URL: {base_url}")
    print()
    
    print("💡 提示:")
    print("   - 设置环境变量 GROK2API_API_KEY 来使用你的密钥")
    print("   - 设置环境变量 GROK2API_BASE_URL 来修改服务地址")
    print()
    
    # 运行所有演示
    try:
        demo_basic_chat()
        demo_system_prompt()
        demo_streaming()
        demo_different_models()
        demo_temperature()
        
        print("=" * 70)
        print("✅ 所有演示完成！")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        print("\n💡 常见问题:")
        print("   1. 确保服务正在运行 (http://localhost:8002)")
        print("   2. 检查是否配置了有效的 Grok Token")
        print("   3. 如果 IP 被拦截，需要配置代理")


if __name__ == "__main__":
    main()









