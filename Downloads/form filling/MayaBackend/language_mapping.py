"""
Language mapping configuration for the application.

This module contains the LANGUAGE_MAPPING dictionary that maps language names
to their corresponding language codes and voice IDs.
"""
from typing import Dict, Any

LANGUAGE_MAPPING_Deepgram: Dict[str, Dict[str, Any]] = {
    "english": {
        "code": "en",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "hindi": {
        "code": "hi",
        "voice_id": "JNaMjd7t4u3EhgkVknn3"
    },
    "german": {
        "code": "de",
        "voice_id": "rAmra0SCIYOxYmRNDSm3"
    },
    "ukrainian": {
        "code": "uk",
        "voice_id": "U4IxWQ3B5B0suleGgLcn"
    },

    "russian": {
        "code": "ru",
        "voice_id": "OowtKaZH9N7iuGbsd00l"
    },
    "italian": {
        "code": "it",
        "voice_id": "MLpDWJvrjFIdb63xbJp8"
    }
    

}

LANGUAGE_MAPPING_Groq: Dict[str, Dict[str, Any]] = {
    "urdu": {
        "code": "ur",
        "voice_id": "JNaMjd7t4u3EhgkVknn3"
    },
    "farsi": {
        "code": "fa",
        "voice_id": "bj1uMlYGikistcXNmFoh"
    },
    "arabic": {
        "code": "ar",
        "voice_id": "qi4PkV9c01kb869Vh7Su"
    },
        "auto": {
        "code": None,
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "spanish": {
        "code": "es",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "french": {
        "code": "fr",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "chinese": {
        "code": "zh",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "japanese": {
        "code": "ja",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "korean": {
        "code": "ko",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    },
    "portuguese": {
        "code": "pt",
        "voice_id": "9BWtsMINqrJLrRacOk9x"
    }
}