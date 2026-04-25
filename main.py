"""
OmniVoice TTS Server
FastAPI server for OmniVoice voice cloning and synthesis
"""
import os
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from config import (
    API_TITLE, API_DESCRIPTION, API_VERSION, HOST, PORT, LOG_LEVEL,
    ALLOWED_AUDIO_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES,
    MODEL_NAME, DEVICE, DTYPE, SUPPORTED_FORMATS
)
from schemas import (
    TTSCloneRequest, TTSDesignRequest, TTSAutoRequest,
    ServerInfo, HealthResponse
)
from tts_service import tts_service


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup: Load model
    print("=" * 50)
    print("Starting OmniVoice TTS Server...")
    print("=" * 50)
    
    success = tts_service.load_model()
    if not success:
        print("WARNING: Model failed to load. TTS endpoints will not work.")
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down OmniVoice TTS Server...")
    tts_service.cleanup_temp_files()


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "OmniVoice Team",
        "url": "https://github.com/k2-fsa/OmniVoice",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Root & Health Endpoints ==========

@app.get("/", tags=["System"], response_model=HealthResponse)
async def root():
    """
    Root endpoint - Service information
    
    Returns basic service status and documentation links.
    """
    return HealthResponse(
        success=True,
        message="OmniVoice TTS Server is running",
        data={
            "service": "OmniVoice TTS Server",
            "version": API_VERSION,
            "model_loaded": tts_service.is_ready(),
            "model": MODEL_NAME,
            "device": DEVICE,
            "docs": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }
    )


@app.get("/health", tags=["System"], response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns detailed health status including model state.
    """
    return HealthResponse(
        success=tts_service.is_ready(),
        message="healthy" if tts_service.is_ready() else "model not loaded",
        data={
            "status": "healthy" if tts_service.is_ready() else "unhealthy",
            "model_loaded": tts_service.is_ready(),
            "device": DEVICE,
            "dtype": DTYPE
        }
    )


@app.get("/api/v1/info", tags=["System"], response_model=ServerInfo)
async def get_info():
    """
    Get server information
    
    Returns detailed server configuration and available endpoints.
    """
    return ServerInfo(
        name="OmniVoice TTS Server",
        version=API_VERSION,
        model=MODEL_NAME,
        device=DEVICE,
        supported_formats=SUPPORTED_FORMATS,
        endpoints=[
            "/ - Service root",
            "/health - Health check",
            "/docs - Swagger UI documentation",
            "/redoc - ReDoc documentation",
            "/api/v1/info - Server information",
            "/api/v1/tts/clone - Voice cloning (URL)",
            "/api/v1/tts/clone/upload - Voice cloning (file upload)",
            "/api/v1/tts/design - Voice design",
            "/api/v1/tts/auto - Auto voice generation"
        ]
    )


# ========== TTS Endpoints ==========

@app.post("/api/v1/tts/clone", tags=["TTS - Voice Cloning"])
async def tts_clone(request: TTSCloneRequest):
    """
    Voice Cloning - URL Mode
    
    Clone voice from reference audio URL.
    
    ## Parameters
    - **text**: Text to synthesize
    - **ref_audio_url**: URL of reference audio file (wav/mp3/flac)
    - **ref_text**: Transcription of reference audio (optional, auto-transcribed if empty)
    - **output_format**: Output format (wav/mp3), default: wav
    - **num_steps**: Diffusion steps (16 for faster, 32 for better quality)
    - **speed**: Speech speed factor (>1.0 faster, <1.0 slower)
    - **duration**: Fixed output duration in seconds (optional)
    
    ## Tips
    - Use 3-10 seconds reference audio for best results
    - Longer reference audio slows inference and may degrade quality
    - For cross-lingual cloning, the result will have accent from reference language
    """
    if not tts_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded"
        )
    
    try:
        # Download reference audio
        ref_path = await tts_service.download_audio(request.ref_audio_url)
        if ref_path is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to download reference audio from URL"
            )
        
        # Generate audio
        audio_array = tts_service.generate_voice_clone(
            text=request.text,
            ref_audio_path=ref_path,
            ref_text=request.ref_text,
            num_steps=request.num_steps,
            speed=request.speed,
            duration=request.duration
        )
        
        # Convert to requested format
        audio_bytes = tts_service.convert_format(
            audio_array, request.output_format
        )
        
        # Cleanup temp files
        try:
            ref_path.unlink()
        except:
            pass
        
        # Return streaming response
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.output_format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.post("/api/v1/tts/clone/upload", tags=["TTS - Voice Cloning"])
async def tts_clone_upload(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio file (wav/mp3/flac/ogg)"),
    ref_text: Optional[str] = Form(None, description="Transcription of reference audio (optional)"),
    output_format: str = Form("wav", description="Output format: wav or mp3"),
    num_steps: int = Form(32, description="Diffusion steps (16-100)"),
    speed: float = Form(1.0, description="Speed factor (0.1-3.0)"),
    duration: Optional[float] = Form(None, description="Fixed duration in seconds (optional)")
):
    """
    Voice Cloning - File Upload Mode
    
    Clone voice by uploading a reference audio file.
    
    ## Parameters (Form Data)
    - **text**: Text to synthesize
    - **ref_audio**: Reference audio file (multipart/form-data)
    - **ref_text**: Transcription (optional)
    - **output_format**: wav or mp3
    - **num_steps**: Diffusion steps
    - **speed**: Speech speed factor
    - **duration**: Fixed duration (optional)
    
    ## Supported Formats
    - WAV (.wav)
    - MP3 (.mp3)
    - FLAC (.flac)
    - OGG (.ogg)
    - M4A (.m4a)
    
    ## Tips
    - Use 3-10 seconds reference audio for best results
    - Maximum file size: {max_size}MB
    """.format(max_size=MAX_FILE_SIZE_MB)
    
    if not tts_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded"
        )
    
    # Validate file extension
    file_ext = Path(ref_audio.filename).suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {file_ext}. Supported: {ALLOWED_AUDIO_EXTENSIONS}"
        )
    
    # Validate output format
    if output_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported output format: {output_format}. Supported: {SUPPORTED_FORMATS}"
        )
    
    try:
        # Read uploaded file
        file_content = await ref_audio.read()
        
        # Save to temp
        ref_path = tts_service.save_uploaded_audio(file_content, ref_audio.filename)
        
        # Generate audio
        audio_array = tts_service.generate_voice_clone(
            text=text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            num_steps=num_steps,
            speed=speed,
            duration=duration
        )
        
        # Convert to requested format
        audio_bytes = tts_service.convert_format(audio_array, output_format)
        
        # Cleanup temp files
        try:
            ref_path.unlink()
        except:
            pass
        
        # Return streaming response
        media_type = "audio/wav" if output_format == "wav" else "audio/mpeg"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{output_format}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.post("/api/v1/tts/clone/local", tags=["TTS - Voice Cloning"])
async def tts_clone_local(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio_path: str = Form(..., description="Absolute path to local reference audio file (wav/mp3/flac/ogg)"),
    ref_text: Optional[str] = Form(None, description="Transcription of reference audio (optional)"),
    output_format: str = Form("wav", description="Output format: wav or mp3"),
    num_steps: int = Form(32, description="Diffusion steps (16-100)"),
    speed: float = Form(1.0, description="Speed factor (0.1-3.0)"),
    duration: Optional[float] = Form(None, description="Fixed duration in seconds (optional)")
):
    """
    Voice Cloning - Local File Path Mode

    Clone voice using a reference audio file from local disk path.
    No file upload needed - directly reads from the provided absolute path.

    ## Parameters (Form Data)
    - **text**: Text to synthesize
    - **ref_audio_path**: Absolute path to local reference audio file
    - **ref_text**: Transcription of reference audio (optional)
    - **output_format**: wav or mp3
    - **num_steps**: Diffusion steps
    - **speed**: Speech speed factor
    - **duration**: Fixed duration (optional)

    ## Supported Formats
    - WAV (.wav)
    - MP3 (.mp3)
    - FLAC (.flac)
    - OGG (.ogg)
    - M4A (.m4a)

    ## Example
    ```
    ref_audio_path: "C:/Users/audio/reference.mp3"
    ```

    ## Security Note
    - Only local file paths are supported
    - Ensure the file exists and is readable
    - The server must have permission to access the file
    """

    if not tts_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded"
        )

    # Convert to Path object and validate
    ref_path = Path(ref_audio_path)

    # Check if path is absolute
    if not ref_path.is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ref_audio_path must be an absolute path"
        )

    # Check if file exists
    if not ref_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reference audio file not found: {ref_audio_path}"
        )

    # Check if it's a file
    if not ref_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a file: {ref_audio_path}"
        )

    # Validate file extension
    file_ext = ref_path.suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {file_ext}. Supported: {ALLOWED_AUDIO_EXTENSIONS}"
        )

    # Validate output format
    if output_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported output format: {output_format}. Supported: {SUPPORTED_FORMATS}"
        )

    try:
        # Generate audio directly from local file path
        audio_array = tts_service.generate_voice_clone(
            text=text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            num_steps=num_steps,
            speed=speed,
            duration=duration
        )

        # Convert to requested format
        audio_bytes = tts_service.convert_format(audio_array, output_format)

        # Return streaming response
        media_type = "audio/wav" if output_format == "wav" else "audio/mpeg"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{output_format}"
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.post("/api/v1/tts/design", tags=["TTS - Voice Design"])
async def tts_design(request: TTSDesignRequest):
    """
    Voice Design
    
    Design voice using speaker attributes without reference audio.
    
    ## Parameters
    - **text**: Text to synthesize
    - **instruct**: Voice attributes (comma-separated)
    - **output_format**: Output format (wav/mp3)
    - **num_steps**: Diffusion steps
    - **speed**: Speech speed factor
    - **duration**: Fixed duration (optional)
    
    ## Voice Attributes
    
    ### Gender
    - `male`, `female`
    
    ### Age
    - `child`, `young adult`, `middle aged`, `elderly`
    
    ### Pitch
    - `very low`, `low`, `moderate`, `high`, `very high`
    
    ### Style
    - `whisper` (whispering voice)
    
    ### English Accents
    - `American`, `British`, `Australian`, `Indian`, `Scottish`, etc.
    
    ### Chinese Dialects
    - `四川话` (Sichuan dialect)
    - `陕西话` (Shaanxi dialect)
    - And more...
    
    ## Examples
    - `"female, low pitch, British accent"`
    - `"male, child, high pitch"`
    - `"female, elderly, whisper"`
    - `"male, young adult, American accent"`
    
    ## Note
    Voice design was trained on Chinese and English data. Results may vary for other languages.
    """
    
    if not tts_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded"
        )
    
    try:
        # Generate audio
        audio_array = tts_service.generate_voice_design(
            text=request.text,
            instruct=request.instruct,
            num_steps=request.num_steps,
            speed=request.speed,
            duration=request.duration
        )
        
        # Convert to requested format
        audio_bytes = tts_service.convert_format(
            audio_array, request.output_format
        )
        
        # Return streaming response
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.output_format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.post("/api/v1/tts/auto", tags=["TTS - Auto Voice"])
async def tts_auto(request: TTSAutoRequest):
    """
    Auto Voice Generation
    
    Generate speech with automatically selected voice.
    The model will choose an appropriate voice based on the text content.
    
    ## Parameters
    - **text**: Text to synthesize
    - **output_format**: Output format (wav/mp3)
    - **num_steps**: Diffusion steps
    - **speed**: Speech speed factor
    - **duration**: Fixed duration (optional)
    
    ## Use Case
    When you don't have a reference audio and don't need specific voice attributes,
    the model will generate speech in a natural, context-appropriate voice.
    """
    
    if not tts_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded"
        )
    
    try:
        # Generate audio
        audio_array = tts_service.generate_auto_voice(
            text=request.text,
            num_steps=request.num_steps,
            speed=request.speed,
            duration=request.duration
        )
        
        # Convert to requested format
        audio_bytes = tts_service.convert_format(
            audio_array, request.output_format
        )
        
        # Return streaming response
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.output_format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}",
            "data": None
        }
    )


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting OmniVoice TTS Server on {HOST}:{PORT}")
    print(f"Documentation: http://{HOST}:{PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=False
    )
