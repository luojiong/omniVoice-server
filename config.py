"""
OmniVoice Server Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "12330"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# OmniVoice model configuration
MODEL_NAME = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.getenv("DEVICE", "cuda:0")  # cuda:0, cuda:1, cpu
DTYPE = os.getenv("DTYPE", "float16")  # float16, float32, bfloat16

# Audio configuration
SAMPLE_RATE = 24000  # OmniVoice outputs at 24kHz
SUPPORTED_FORMATS = ["wav", "mp3"]
DEFAULT_FORMAT = "wav"

# TTS generation defaults
DEFAULT_NUM_STEPS = 32  # Diffusion steps (16 for faster, 32 for better quality)
DEFAULT_SPEED = 1.0  # Speed factor (>1.0 faster, <1.0 slower)
DEFAULT_DURATION = None  # Fixed duration in seconds (None for auto)

# Upload configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"}

# API documentation
API_TITLE = "OmniVoice TTS Server"
API_DESCRIPTION = """
OmniVoice 语音合成服务 API

## 功能

### 1. 语音克隆 (Voice Cloning)
通过参考音频克隆说话人的声音特征

### 2. 声音设计 (Voice Design)
通过描述声音属性（性别、年龄、音调、口音等）设计声音

### 3. 自动声音 (Auto Voice)
自动生成符合文本语义的随机声音

## 技术规格

- **支持语言**: 600+ 种语言
- **音频采样率**: 24kHz
- **输出格式**: WAV / MP3
- **推理速度**: 最快可达 40x 实时速度

## 模型

基于扩散语言模型 (Diffusion Language Model) 架构
"""
API_VERSION = "1.0.0"

# CORS (for internal use, restrict as needed)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
