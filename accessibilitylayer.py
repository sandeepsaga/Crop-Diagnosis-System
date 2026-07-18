import os
import logging
import edge_tts
from deep_translator import GoogleTranslator

logger = logging.getLogger("accessibility")

# Map of standard state/location contexts to primary regional spoken languages
LANGUAGE_MAP = {
    "odisha": {"code": "or", "voice": "or-IN-SubhasiniNeural"},  # Odia
    "hindi": {"code": "hi", "voice": "hi-IN-SwaraNeural"},      # Hindi (Fallback default)
}

def translate_treatment_plan(text: str, state_context: str = "odisha") -> str:
    """
    Translates the structural English treatment plan into the target local language.
    """
    target = state_context.lower().strip()
    lang_code = LANGUAGE_MAP.get(target, LANGUAGE_MAP["hindi"])["code"]
    
    try:
        logger.info(f"Translating advice report to local language code: '{lang_code}'...")
        translated_text = GoogleTranslator(source='auto', target=lang_code).translate(text)
        return translated_text
    except Exception as e:
        logger.error(f"Translation engine dropped out: {str(e)}. Defaulting to raw text.")
        return text

async def generate_voice_note(text: str, output_filename: str, state_context: str = "odisha") -> str:
    """
    Converts local language text into a highly natural audio speech file (.mp3).
    Returns the absolute path to the generated asset.
    """
    target = state_context.lower().strip()
    voice_profile = LANGUAGE_MAP.get(target, LANGUAGE_MAP["hindi"])["voice"]
    
    # Ensure a local static directory exists to hold public media files
    os.makedirs("static", exist_ok=True)
    file_path = os.path.join("static", output_filename)
    
    try:
        logger.info(f"Generating localized audio payload using voice: {voice_profile}...")
        
        # Target path diagnostics
        abs_path = os.path.abspath(file_path)
        logger.info(f"DEBUG [Point 3]: Target Absolute Storage Path is: {abs_path}")
        
        communicate = edge_tts.Communicate(text, voice_profile)
        await communicate.save(file_path)
        
        # Verify if the file physically exists right after saving
        if os.path.exists(file_path):
            logger.info(f"Audio asset successfully compiled and verified at: {file_path}")
        else:
            logger.error(f"Write complete but file missing from disk layout at: {file_path}")
            
        return file_path
    except Exception as e:
        logger.error(f"TTS compilation failure: {str(e)}")
        raise e