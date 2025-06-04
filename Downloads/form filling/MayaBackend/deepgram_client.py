import asyncio
import logging
import os
from time import perf_counter
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Deepgram SDK
try:
    from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions
except ImportError:
    logger.error("Deepgram SDK not installed. Please install with 'pip install deepgram-sdk'")

# Default Deepgram API key, should be replaced with environment variable
DEEPGRAM_API_KEY = "<DEEPGRAM_API_KEY>"

# Singleton client
_deepgram_client = None

def initialize_deepgram(api_key: Optional[str] = None):
    """Initialize the Deepgram client with the provided API key"""
    global DEEPGRAM_API_KEY, _deepgram_client
    
    # Use provided API key or fall back to environment variable
    DEEPGRAM_API_KEY = api_key or DEEPGRAM_API_KEY or os.environ.get("DEEPGRAM_API_KEY")
    
    if not DEEPGRAM_API_KEY:
        logger.warning("DEEPGRAM_API_KEY not set. Please set the DEEPGRAM_API_KEY environment variable.")
        return None
    
    try:
        # Create client options with keepalive enabled for faster API calls
        options = DeepgramClientOptions(
            # Note: in v3 SDK, the first parameter to DeepgramClient is the API key directly
            # The api_key is not set here, but passed as first parameter to DeepgramClient
            options={
                "keepalive": True,  # Enable connection keepalive for faster API calls
                "keep_alive": True  # Alternative spelling also set for compatibility
            }
        )
        
        # Create a new client with optimized connection settings
        # In v3, the API key is passed as the first parameter to DeepgramClient
        _deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, options)
        logger.info(f"Initialized Deepgram client with API key: {DEEPGRAM_API_KEY[:5]}... and keepalive enabled")
        
        return _deepgram_client
    except Exception as e:
        logger.error(f"Error initializing Deepgram client: {e}")
        return None

def get_deepgram_client():
    """Get the initialized Deepgram client or create a new one"""
    global _deepgram_client
    
    if _deepgram_client is None:
        _deepgram_client = initialize_deepgram()
    
    return _deepgram_client

async def transcribe_audio_file(file_obj, language: Optional[str] = None, model: str = "nova-3") -> Dict[str, Any]:
    """Transcribe audio using Deepgram
    
    Args:
        file_obj: File-like object containing audio data
        language: Optional language code
        model: Deepgram model to use (default: nova-3)
        
    Returns:
        Dict with transcription results
    """
    try:
        # Start API call timing
        start_time_api = perf_counter()
        
        # Get Deepgram client
        deepgram = get_deepgram_client()
        if not deepgram:
            return {"error": "Deepgram client not initialized", "text": "Unable to transcribe audio. Please try again."}
        
        # Create options for transcription
        options = PrerecordedOptions(
            model=model,
            smart_format=True,  # Adds punctuation and capitalization
            diarize=False,      # We don't need speaker diarization for this application
        )
        
        # Handle language settings
        if language is None or language == "auto":
            # If language is None or explicitly set to "auto", enable auto-detection
            options.detect_language = True
            logger.info("Using automatic language detection with Deepgram")
        else:
            # Otherwise use the specific language code
            options.detect_language = False
            options.language = language
            logger.info(f"Using specified language code with Deepgram: {language}")
        
        logger.info(f"Starting Deepgram transcription with model {model}")
        
        try:
            # Read the file data if it's a file-like object
            if hasattr(file_obj, 'read'):
                file_data = file_obj.read()
            else:
                # Assume it's already binary data
                file_data = file_obj
                
            # Create the payload with binary data
            payload = {"buffer": file_data}
            
            # Use the REST API with transcribe_file - Note: v3 SDK doesn't use await for this method
            # It returns a response object directly, not a coroutine
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
            
            # Calculate API call time
            api_time = perf_counter() - start_time_api
            
            # Extract the transcription text from the response
            if response and hasattr(response, "results") and hasattr(response.results, "channels"):
                # Get the transcript from the first channel alternative
                transcript = response.results.channels[0].alternatives[0].transcript
                logger.info(f"Deepgram transcription completed in {api_time:.3f}s")
                logger.info(f"Transcription received: {transcript}")
                
                # Return a dict in the same format as Whisper for compatibility
                return {"text": transcript}
            else:
                logger.error(f"Unexpected Deepgram response format after {api_time:.3f}s")
                return {"error": "Invalid response format", "text": "Unable to transcribe audio. Please try again."}
                
        except Exception as inner_e:
            logger.error(f"Error during Deepgram API call: {inner_e}")
            return {"error": str(inner_e), "text": "Unable to transcribe audio. Please try again."}
                
    except Exception as e:
        logger.error(f"Error in Deepgram transcription: {e}")
        return {"error": str(e), "text": "Unable to transcribe audio. Please try again."}
