"""
OmniVoice TTS Service
Handles model loading and audio generation
"""
import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional, Union, List
import aiohttp
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

from config import (
    MODEL_NAME, DEVICE, DTYPE, SAMPLE_RATE,
    TEMP_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_AUDIO_EXTENSIONS
)


class TTSService:
    """OmniVoice TTS Service singleton"""
    
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.device = DEVICE
        self.dtype = getattr(torch, DTYPE)
    
    def load_model(self) -> bool:
        """
        Load OmniVoice model
        
        Returns:
            bool: Whether model loaded successfully
        """
        if self.model is not None:
            return True
        
        try:
            # Import here to avoid loading at module level
            from omnivoice import OmniVoice
            
            print(f"Loading OmniVoice model: {MODEL_NAME}")
            print(f"Device: {self.device}, Dtype: {DTYPE}")
            
            self.model = OmniVoice.from_pretrained(
                MODEL_NAME,
                device_map=self.device,
                dtype=self.dtype
            )
            
            print("OmniVoice model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    async def download_audio(self, url: str) -> Optional[Path]:
        """
        Download audio file from URL
        
        Args:
            url: Audio file URL
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return None
                    
                    content = await response.read()
                    
                    if len(content) > MAX_FILE_SIZE_BYTES:
                        raise ValueError(f"File too large: {len(content)} bytes")
                    
                    # Determine file extension from Content-Type or URL
                    content_type = response.headers.get('Content-Type', '')
                    if 'wav' in content_type:
                        ext = '.wav'
                    elif 'mp3' in content_type:
                        ext = '.mp3'
                    elif 'mpeg' in content_type:
                        ext = '.mp3'
                    else:
                        ext = Path(url).suffix or '.wav'
                    
                    temp_path = TEMP_DIR / f"ref_{os.urandom(8).hex()}{ext}"
                    temp_path.write_bytes(content)
                    
                    return temp_path
                    
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None
    
    def save_uploaded_audio(self, file_content: bytes, filename: str) -> Path:
        """
        Save uploaded audio file
        
        Args:
            file_content: File content bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")
        
        if len(file_content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File too large: {len(file_content)} bytes")
        
        temp_path = TEMP_DIR / f"upload_{os.urandom(8).hex()}{ext}"
        temp_path.write_bytes(file_content)
        
        return temp_path
    
    def convert_to_wav(self, audio_path: Path) -> Path:
        """
        Convert audio to WAV format (24kHz mono)
        
        Args:
            audio_path: Path to input audio
            
        Returns:
            Path to WAV file
        """
        if audio_path.suffix.lower() == '.wav':
            # Check if already correct format
            try:
                info = sf.info(audio_path)
                if info.samplerate == SAMPLE_RATE and info.channels == 1:
                    return audio_path
            except:
                pass
        
        # Convert using pydub
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        wav_path = TEMP_DIR / f"converted_{os.urandom(8).hex()}.wav"
        audio.export(wav_path, format="wav")
        
        return wav_path
    
    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: Optional[str] = None,
        num_steps: int = 32,
        speed: float = 1.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate speech using voice cloning
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio
            ref_text: Transcription of reference audio (optional)
            num_steps: Diffusion steps
            speed: Speed factor
            duration: Fixed duration
            
        Returns:
            Audio array (24kHz)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to WAV if needed
        wav_path = self.convert_to_wav(ref_audio_path)
        
        # Build kwargs
        kwargs = {
            "text": text,
            "ref_audio": str(wav_path),
            "num_step": num_steps,
            "speed": speed,
        }
        
        if ref_text:
            kwargs["ref_text"] = ref_text
        if duration:
            kwargs["duration"] = duration
        
        # Generate audio
        audio = self.model.generate(**kwargs)
        
        return audio[0]  # Return first (and usually only) result
    
    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        num_steps: int = 32,
        speed: float = 1.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate speech using voice design
        
        Args:
            text: Text to synthesize
            instruct: Voice attributes instruction
            num_steps: Diffusion steps
            speed: Speed factor
            duration: Fixed duration
            
        Returns:
            Audio array (24kHz)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        kwargs = {
            "text": text,
            "instruct": instruct,
            "num_step": num_steps,
            "speed": speed,
        }
        
        if duration:
            kwargs["duration"] = duration
        
        audio = self.model.generate(**kwargs)
        return audio[0]
    
    def generate_auto_voice(
        self,
        text: str,
        num_steps: int = 32,
        speed: float = 1.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate speech using auto voice
        
        Args:
            text: Text to synthesize
            num_steps: Diffusion steps
            speed: Speed factor
            duration: Fixed duration
            
        Returns:
            Audio array (24kHz)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        kwargs = {
            "text": text,
            "num_step": num_steps,
            "speed": speed,
        }
        
        if duration:
            kwargs["duration"] = duration
        
        audio = self.model.generate(**kwargs)
        return audio[0]
    
    def convert_format(
        self,
        audio_array: np.ndarray,
        output_format: str,
        sample_rate: int = SAMPLE_RATE
    ) -> bytes:
        """
        Convert audio array to specified format
        
        Args:
            audio_array: Numpy array of audio samples
            output_format: 'wav' or 'mp3'
            sample_rate: Audio sample rate
            
        Returns:
            Audio bytes
        """
        if output_format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            return buffer.getvalue()
        
        elif output_format == "mp3":
            # First save as WAV temp
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_temp:
                sf.write(wav_temp.name, audio_array, sample_rate, subtype="PCM_16")
                wav_path = wav_temp.name
            
            # Convert to MP3
            audio = AudioSegment.from_wav(wav_path)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="192k")
            buffer.seek(0)
            
            # Clean up temp file
            os.unlink(wav_path)
            
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age in hours
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in TEMP_DIR.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                    except:
                        pass


# Global service instance
tts_service = TTSService()
