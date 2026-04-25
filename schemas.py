"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from config import DEFAULT_NUM_STEPS, DEFAULT_SPEED, SUPPORTED_FORMATS


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")


class TTSCloneRequest(BaseModel):
    """
    Voice cloning request
    
    Clone voice from reference audio URL
    """
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of voice cloning."]
    )
    ref_audio_url: str = Field(
        ...,
        description="URL of the reference audio file (wav/mp3/flac)",
        examples=["https://example.com/ref.wav"]
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcription of reference audio (optional, will auto-transcribe if not provided)",
        examples=["This is the reference audio transcription."]
    )
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Output audio format"
    )
    num_steps: int = Field(
        default=DEFAULT_NUM_STEPS,
        description="Diffusion steps (16 for faster, 32 for better quality)",
        ge=1,
        le=100
    )
    speed: float = Field(
        default=DEFAULT_SPEED,
        description="Speech speed factor (>1.0 faster, <1.0 slower)",
        ge=0.1,
        le=3.0
    )
    duration: Optional[float] = Field(
        default=None,
        description="Fixed output duration in seconds (overrides speed)",
        ge=1.0,
        le=60.0
    )


class TTSCloneUploadRequest(BaseModel):
    """
    Voice cloning request with file upload (form data)
    
    This model is used for documentation purposes.
    Actual parameters are received via Form data.
    """
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of voice cloning."]
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcription of reference audio (optional, will auto-transcribe if not provided)",
        examples=["This is the reference audio transcription."]
    )
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Output audio format"
    )
    num_steps: int = Field(
        default=DEFAULT_NUM_STEPS,
        description="Diffusion steps",
        ge=1,
        le=100
    )
    speed: float = Field(
        default=DEFAULT_SPEED,
        description="Speech speed factor",
        ge=0.1,
        le=3.0
    )
    duration: Optional[float] = Field(
        default=None,
        description="Fixed output duration in seconds",
        ge=1.0,
        le=60.0
    )


class TTSDesignRequest(BaseModel):
    """
    Voice design request
    
    Design voice using speaker attributes without reference audio
    """
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of voice design."]
    )
    instruct: str = Field(
        ...,
        description="Voice attributes (comma-separated): gender, age, pitch, style, accent, dialect",
        examples=["female, low pitch, British accent", "male, child, high pitch, whisper"]
    )
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Output audio format"
    )
    num_steps: int = Field(
        default=DEFAULT_NUM_STEPS,
        description="Diffusion steps (16 for faster, 32 for better quality)",
        ge=1,
        le=100
    )
    speed: float = Field(
        default=DEFAULT_SPEED,
        description="Speech speed factor (>1.0 faster, <1.0 slower)",
        ge=0.1,
        le=3.0
    )
    duration: Optional[float] = Field(
        default=None,
        description="Fixed output duration in seconds (overrides speed)",
        ge=1.0,
        le=60.0
    )


class TTSAutoRequest(BaseModel):
    """
    Auto voice request
    
    Generate speech with automatically selected voice
    """
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of automatic voice generation."]
    )
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        description="Output audio format"
    )
    num_steps: int = Field(
        default=DEFAULT_NUM_STEPS,
        description="Diffusion steps",
        ge=1,
        le=100
    )
    speed: float = Field(
        default=DEFAULT_SPEED,
        description="Speech speed factor",
        ge=0.1,
        le=3.0
    )
    duration: Optional[float] = Field(
        default=None,
        description="Fixed output duration in seconds",
        ge=1.0,
        le=60.0
    )


class TTSResponse(BaseResponse):
    """TTS generation response metadata"""
    data: Optional[dict] = Field(
        default=None,
        description="Response data including audio URL or base64"
    )


class ServerInfo(BaseModel):
    """Server information response"""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    model: str = Field(..., description="TTS model name")
    device: str = Field(..., description="Compute device")
    supported_formats: List[str] = Field(..., description="Supported output formats")
    endpoints: List[str] = Field(..., description="Available API endpoints")


class HealthResponse(BaseResponse):
    """Health check response"""
    data: Optional[dict] = Field(
        default=None,
        description="Health status details"
    )
