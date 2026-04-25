#!/usr/bin/env python3
"""
OmniVoice TTS 本地路径模式测试脚本

测试流程：
1. 检查TTS服务健康状态
2. 语音克隆测试 - 使用本地文件路径方式读取参考音频

运行前请确保：
- TTS服务已启动: uv run python main.py
- 资源文件已准备: resource/test.mp3, resource/test.txt

使用方法:
uv run python tests/test_local.py
"""

import sys
import io
import time
import wave
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    print("错误: 缺少 requests 依赖")
    print("请安装: uv add requests")
    sys.exit(1)


# ============== 配置 ==============
BASE_URL = "http://localhost:12330"
RESOURCE_DIR = Path(__file__).parent / "resource"
OUTPUT_DIR = Path(__file__).parent / "output"

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== 工具函数 ==============

def log_info(message: str) -> None:
    """打印信息日志"""
    print(f"[INFO] {message}")


def log_error(message: str) -> None:
    """打印错误日志"""
    print(f"[ERROR] {message}", file=sys.stderr)


def log_success(message: str) -> None:
    """打印成功日志"""
    print(f"[OK] {message}")


def generate_output_filename(prefix: str) -> Path:
    """
    生成输出文件名

    格式: {prefix}_YYYYMMDD_HHMMSS.wav
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.wav"
    return OUTPUT_DIR / filename


def read_test_text() -> str:
    """
    读取测试文本

    从 resource/test.txt 读取完整内容作为测试文本

    Returns:
        测试文本内容

    Raises:
        SystemExit: 如果文件不存在或读取失败
    """
    text_file = RESOURCE_DIR / "test.txt"

    if not text_file.exists():
        log_error(f"测试文本文件不存在: {text_file}")
        log_info("请确保 resource/test.txt 文件存在")
        sys.exit(1)

    try:
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
            else:
                log_error(f"测试文本文件为空: {text_file}")
                sys.exit(1)

    except Exception as e:
        log_error(f"读取测试文本失败: {e}")
        sys.exit(1)


def check_service_health() -> bool:
    """
    检查TTS服务健康状态

    发送请求到 /health 检查服务是否就绪

    Returns:
        True if service is healthy

    Raises:
        SystemExit: 如果服务未启动或异常
    """
    log_info("检查TTS服务健康状态...")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                log_success(f"服务健康: {BASE_URL}")
                log_info(f"模型加载: {'是' if data.get('data', {}).get('model_loaded') else '否'}")
                log_info(f"运行设备: {data.get('data', {}).get('device', 'unknown')}")
                return True
            else:
                log_error(f"服务未就绪: {data.get('message', 'unknown')}")
                sys.exit(1)
        else:
            log_error(f"服务返回异常状态码: {response.status_code}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        log_error(f"无法连接到TTS服务: {BASE_URL}")
        log_info("请确保TTS服务已启动: uv run python main.py")
        sys.exit(1)

    except requests.exceptions.Timeout:
        log_error("服务响应超时")
        sys.exit(1)

    except Exception as e:
        log_error(f"检查服务时发生错误: {e}")
        sys.exit(1)


def test_voice_clone_local() -> Tuple[Path, float, float]:
    """
    测试语音克隆功能 - 本地文件路径模式

    使用本地磁盘绝对路径直接读取参考音频进行语音克隆

    Returns:
        Tuple[生成的音频文件路径, 生成耗时秒数, 音频时长秒数]

    Raises:
        SystemExit: 如果测试失败
    """
    print("\n" + "=" * 50)
    log_info("测试: 语音克隆 - 本地文件路径模式 (Voice Cloning - Local Path)")
    print("=" * 50)

    # 准备资源 - 使用绝对路径
    ref_audio_path = (RESOURCE_DIR / "test.mp3").resolve()
    if not ref_audio_path.exists():
        log_error(f"参考音频不存在: {ref_audio_path}")
        log_info("请确保 resource/test.mp3 文件存在")
        sys.exit(1)

    # 读取测试文本
    test_text = read_test_text()
    log_info(f"合成文本: {test_text[:50]}..." if len(test_text) > 50 else f"合成文本: {test_text}")
    log_info(f"参考音频路径: {ref_audio_path}")

    # 准备请求 - 使用本地路径接口
    url = f"{BASE_URL}/api/v1/tts/clone/local"

    try:
        # 准备表单数据
        data = {
            "text": test_text,
            "ref_audio_path": str(ref_audio_path),
            "ref_text": "你好,这是一条测试语句",
            "output_format": "wav",
            "num_steps": "32",
            "speed": "1.5"
        }

        log_info("正在发送请求...")
        start_time = time.time()
        response = requests.post(url, data=data)
        elapsed_time = time.time() - start_time

        # 处理响应
        if response.status_code == 200:
            # 保存生成的音频
            output_path = generate_output_filename("tts_clone_local")
            with open(output_path, "wb") as f:
                f.write(response.content)

            # 计算音频时长
            audio_duration = 0.0
            try:
                with wave.open(str(output_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    audio_duration = frames / float(rate)
            except Exception:
                pass

            log_success(f"音频生成成功")
            log_info(f"输出文件: {output_path}")
            log_info(f"文件大小: {len(response.content) / 1024:.1f} KB")
            log_info(f"生成耗时: {elapsed_time:.2f} 秒")
            log_info(f"音频时长: {audio_duration:.2f} 秒")

            return output_path, elapsed_time, audio_duration

        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "unknown error")
            except:
                error_msg = response.text or f"HTTP {response.status_code}"

            log_error(f"语音克隆失败: {error_msg}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        log_error("连接服务失败")
        sys.exit(1)

    except Exception as e:
        log_error(f"语音克隆测试失败: {e}")
        sys.exit(1)


def print_summary(clone_file: Path, elapsed_time: float, audio_duration: float) -> None:
    """
    打印测试总结

    Args:
        clone_file: 语音克隆生成的文件路径
        elapsed_time: 生成耗时（秒）
        audio_duration: 音频时长（秒）
    """
    print("\n" + "=" * 50)
    log_info("测试完成")
    print("=" * 50)
    print(f"\n生成文件:")
    print(f" 语音克隆: {clone_file}")
    print(f"\n统计信息:")
    print(f" 生成耗时: {elapsed_time:.2f} 秒")
    print(f" 音频时长: {audio_duration:.2f} 秒")
    print(f" RTF: {elapsed_time/audio_duration:.2f}" if audio_duration > 0 else "")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"\n你可以使用以下命令播放生成的音频:")
    print(f" ffplay {clone_file.name}")
    print()


# ============== 主入口 ==============

def main() -> None:
    """
    主入口函数

    执行完整的测试流程
    """
    print("=" * 50)
    print("OmniVoice TTS 本地路径模式测试")
    print("=" * 50)

    # 检查资源目录
    if not RESOURCE_DIR.exists():
        log_error(f"资源目录不存在: {RESOURCE_DIR}")
        sys.exit(1)

    # 1. 检查服务
    check_service_health()

    # 2. 测试语音克隆（本地路径模式）
    clone_output, elapsed_time, audio_duration = test_voice_clone_local()

    # 3. 打印总结
    print_summary(clone_output, elapsed_time, audio_duration)

    # 正常退出
    sys.exit(0)


if __name__ == "__main__":
    main()
