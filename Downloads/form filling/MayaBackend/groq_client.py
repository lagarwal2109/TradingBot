import os
import time
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq API Key - Consider using environment variable for production
GROQ_API_KEY = "<GROQ_API_KEY>"

# Global variable to hold the AsyncClient instance
_http_client = None

def get_async_http_client():
    """Returns a reusable httpx.AsyncClient with optimized connection settings"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
            timeout=60.0
        )
    return _http_client

async def transcribe_audio_groq(file_data, model="whisper-large-v3", language=None):
    """
    Transcribes audio using Groq's API
    
    Args:
        file_data (bytes): Binary audio data
        model (str): Model to use for transcription
        language (str): Language code (e.g., 'en', 'de', 'fr')
        
    Returns:
        str: The transcribed text
    """
    client = get_async_http_client()
    start_time = time.time()
    
    try:
        logger.info(f"Starting audio transcription with Groq API")
        
        # Directly use the binary data
        files = {"file": ("audio.webm", file_data)}
        
        # Set up language parameter
        if language == None:
            data = {"model": model}
        else:
            data = {"model": model, "language": language}
            
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=data,
            files=files,
            headers=headers
        )
        
        duration = time.time() - start_time
        logger.info(f"Groq transcription completed in {duration:.2f} seconds")
        
        response.raise_for_status()
        result = response.json()
        return result.get("text")
    
    except Exception as e:
        logger.error(f"Error during Groq transcription: {str(e)}")
        raise
