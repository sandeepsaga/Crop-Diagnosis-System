import os
import json
import logging
import re
from gtts import gTTS
import edge_tts
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("accessibility")

# This module makes its own Gemini call, so it needs its own client instance
# (main.py's vision_client is not visible here — separate module/process scope).
_translation_key = os.environ.get("GEMINI_API_KEY")
_translation_client = genai.Client(api_key=_translation_key)


class TreatmentTranslation(BaseModel):
    display_text: str = Field(description="WhatsApp-formatted, localized treatment plan text, using single *asterisk* bold only, no markdown headers or dividers.")
    speech_text: str = Field(description="TTS-ready localized treatment plan text, all units/currency/temperature fully spelled out in words, no markdown symbols at all.")

# ---------------------------------------------------------------------------
# Language coverage: map Indian states/UTs -> a language key.
# Extend this as you onboard more regions; unmapped states fall back to Hindi,
# which is the most broadly understood option across most of India.
# ---------------------------------------------------------------------------
STATE_TO_LANGUAGE = {
    "odisha": "odia",
    "west bengal": "bengali",
    "tamil nadu": "tamil",
    "andhra pradesh": "telugu",
    "telangana": "telugu",
    "maharashtra": "marathi",
    "gujarat": "gujarati",
    "karnataka": "kannada",
    "kerala": "malayalam",
    "punjab": "punjabi",
    "uttar pradesh": "hindi",
    "bihar": "hindi",
    "madhya pradesh": "hindi",
    "rajasthan": "hindi",
    "haryana": "hindi",
    "delhi": "hindi",
    "jharkhand": "hindi",
    "chhattisgarh": "hindi",
    "uttarakhand": "hindi",
    "himachal pradesh": "hindi",
}

# Per-language voice/engine configuration.
# gtts_lang = None means gTTS has no usable voice for that language, so the
# fallback path will render a generic spoken notice instead of silent-failing.
LANGUAGE_MAP = {
    # voice=None means Edge TTS has no native voice for this language (verified
    # against `edge-tts --list-voices`). generate_voice_note() will skip audio
    # generation entirely for these rather than sending a misleading fallback.
    "odia":       {"code": "or", "voice": None,                    "gtts_lang": None},
    "punjabi":    {"code": "pa", "voice": None,                    "gtts_lang": "pa"},
    "hindi":      {"code": "hi", "voice": "hi-IN-SwaraNeural",     "gtts_lang": "hi"},
    "bengali":    {"code": "bn", "voice": "bn-IN-TanishaaNeural",  "gtts_lang": "bn"},
    "tamil":      {"code": "ta", "voice": "ta-IN-PallaviNeural",   "gtts_lang": "ta"},
    "telugu":     {"code": "te", "voice": "te-IN-ShrutiNeural",    "gtts_lang": "te"},
    "marathi":    {"code": "mr", "voice": "mr-IN-AarohiNeural",    "gtts_lang": "mr"},
    "gujarati":   {"code": "gu", "voice": "gu-IN-DhwaniNeural",    "gtts_lang": "gu"},
    "kannada":    {"code": "kn", "voice": "kn-IN-SapnaNeural",     "gtts_lang": "kn"},
    "malayalam":  {"code": "ml", "voice": "ml-IN-SobhanaNeural",   "gtts_lang": "ml"},
}

# Languages with NO working TTS voice on either engine right now.
# generate_voice_note() returns None for these — main.py should skip the
# audio-dispatch step rather than sending a misleading placeholder file.
NO_TTS_SUPPORT = {"odia"}

# Short spoken notice used only if cleaned_text ends up empty, per language.
# Falls back to Hindi if a language isn't listed here.
FALLBACK_MESSAGES = {
    "odia": "ଆପଣଙ୍କ ଫସଲର ରିପୋର୍ଟ ପ୍ରସ୍ତୁତ ହୋଇଯାଇଛି।",
    "hindi": "आपकी फसल की रिपोर्ट तैयार हो गई है।",
    "bengali": "আপনার ফসলের রিপোর্ট প্রস্তুত হয়ে গেছে।",
    "tamil": "உங்கள் பயிர் அறிக்கை தயாராகிவிட்டது.",
    "telugu": "మీ పంట నివేదిక సిద్ధమైంది.",
    "marathi": "तुमचा पीक अहवाल तयार झाला आहे.",
    "gujarati": "તમારો પાક અહેવાલ તૈયાર થઈ ગયો છે.",
    "kannada": "ನಿಮ್ಮ ಬೆಳೆ ವರದಿ ಸಿದ್ಧವಾಗಿದೆ.",
    "malayalam": "നിങ്ങളുടെ വിള റിപ്പോർട്ട് തയ്യാറായി.",
    "punjabi": "ਤੁਹਾਡੀ ਫਸਲ ਦੀ ਰਿਪੋਰਟ ਤਿਆਰ ਹੋ ਗਈ ਹੈ।",
}


def _resolve_language_key(state_context: str) -> str:
    """Map a free-text state name to a language key, defaulting to Hindi."""
    normalized = state_context.lower().strip()
    return STATE_TO_LANGUAGE.get(normalized, "hindi")


def translate_treatment_plan(treatment_plan, state_context: str) -> tuple[str, str]:
    """
    Localizes the treatment plan into the primary language spoken in
    state_context, returning (display_text, speech_text).

    display_text: WhatsApp-formatted (single *asterisk* bold), symbols like
    32°C / ₹500 / 5kg left as normal written form since it's read visually.

    speech_text: fully spelled out in words (units, currency, temperature)
    in that language's natural spoken form, no markdown symbols — this is
    what TTS will read, so nothing here should rely on post-hoc regex swaps.
    """
    prompt = f"""
Translate this agricultural treatment plan into the primary language spoken in
{state_context}, India (e.g. Odia for Odisha, Tamil for Tamil Nadu, Marathi for
Maharashtra, etc. — infer the correct language from the state name).

Return JSON with two keys:
- "display_text": for WhatsApp chat. Use WhatsApp formatting only (single
  *asterisk* for bold). Numbers/units/currency can stay in normal written form
  (32°C, ₹500, 5kg) since this is read visually.
- "speech_text": for text-to-speech narration in the same language. No markdown
  symbols at all. Every number, unit, currency amount, and temperature must be
  fully spelled out as words in that language, following that language's natural
  spoken conventions (grammar, gender agreement, etc.), since a TTS engine reads
  this literally and cannot pronounce symbols correctly.

Treatment plan:
{treatment_plan}
"""
    response = _translation_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TreatmentTranslation,
            temperature=0.1
        )
    )

    try:
        result = TreatmentTranslation.model_validate_json(response.text)
        return result.display_text, result.speech_text
    except Exception as parse_err:
        # Should be rare with response_schema enforcing constrained decoding,
        # but log the raw payload so a malformed case can actually be debugged
        # instead of just crashing the whole background worker silently.
        logger.error(f"Failed to parse translation JSON: {parse_err}")
        logger.error(f"Raw response text was:\n{response.text}")
        raise


async def generate_voice_note(text: str, output_filename: str, state_context: str = "odisha") -> str:
    """
    Converts text to an .mp3 voice note using Edge-TTS, falling back gracefully
    to gTTS (or a short spoken notice) to prevent pipeline failure.

    `text` is expected to already be the LLM-produced speech_text — i.e. units,
    currency, and temperature already spelled out in words for the target
    language. This function only strips residual markdown/punctuation symbols;
    it does not perform any language-specific word substitution.
    """
    lang_key = _resolve_language_key(state_context)
    config = LANGUAGE_MAP.get(lang_key, LANGUAGE_MAP["hindi"])
    voice_profile = config["voice"]
    gtts_lang = config["gtts_lang"]

    if lang_key in NO_TTS_SUPPORT:
        logger.warning(
            f"No working TTS voice exists for '{lang_key}' on either engine. "
            f"Skipping audio generation — text report will stand alone for this farmer."
        )
        return None

    os.makedirs("static", exist_ok=True)
    file_path = os.path.join("static", output_filename)

    # Language-agnostic punctuation/symbol scrubbing only.
    # (No hardcoded word substitutions here — speech_text already arrives
    # pre-localized per language from translate_treatment_plan.)
    cleaned_text = text
    cleaned_text = cleaned_text.replace("|", " ")
    cleaned_text = cleaned_text.replace("।", " . ")  # traditional danda -> spoken pause
    cleaned_text = re.sub(r"#{1,6}\s*", "", cleaned_text)
    cleaned_text = re.sub(r"-\s*-\s*-\s*", "", cleaned_text)
    cleaned_text = re.sub(r"[\*#_`\-─│┌┐└┘├┤┬┴┼:!,?]", " ", cleaned_text)
    cleaned_text = cleaned_text.replace("[", "").replace("]", "")

    cleaned_text = re.sub(r"\n+", " . \n", cleaned_text)
    cleaned_text = re.sub(r" +", " ", cleaned_text).strip()

    if not cleaned_text:
        cleaned_text = FALLBACK_MESSAGES.get(lang_key, FALLBACK_MESSAGES["hindi"])

    # Strategy A: Edge TTS (primary, higher quality, widest voice coverage)
    # Only attempt this if a real edge-tts voice actually exists for this
    # language — otherwise skip straight to Strategy B rather than eating a
    # guaranteed, known failure.
    edge_err = None
    if voice_profile is not None:
        try:
            logger.info(f"Attempting Edge TTS generation via voice target: {voice_profile}...")
            communicate = edge_tts.Communicate(cleaned_text, voice_profile)
            await communicate.save(file_path)

            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Edge TTS compilation successful and verified: {file_path}")
                return file_path
            else:
                raise Exception("Edge TTS generated an empty payload file structure.")

        except Exception as e:
            edge_err = e
            logger.warning(f"Primary Edge TTS dropped out ({str(e)}). Activating emergency fallback...")
    else:
        logger.info(f"No edge-tts voice available for '{lang_key}'; going straight to gTTS fallback.")

    # Strategy B: gTTS fallback, or a short generic spoken notice if the
    # language isn't supported by gTTS at all.
    try:
        if gtts_lang:
            logger.info("Compiling speech file via localized gTTS framework...")
            tts = gTTS(text=cleaned_text, lang=gtts_lang, slow=False)
            tts.save(file_path)
        else:
            logger.info("Target language lacks native gTTS support. Rendering safety notification audio asset...")
            safety_text = "Your crop diagnosis report has been successfully generated. Please check your text messages."
            tts = gTTS(text=safety_text, lang="en", slow=False)
            tts.save(file_path)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Emergency safety audio asset compile complete: {file_path}")
            return file_path
        else:
            raise RuntimeError("Backup TTS system returned a dead size metrics envelope.")
    except Exception as fallback_err:
        logger.error(f"Critical Layer Exhaustion: {str(fallback_err)}")
        raise fallback_err
# import os
# import json
# import logging
# import re
# from gtts import gTTS
# import edge_tts
# from google import genai
# from google.genai import types

# logger = logging.getLogger("accessibility")

# # This module makes its own Gemini call, so it needs its own client instance
# # (main.py's vision_client is not visible here — separate module/process scope).
# _translation_key = os.environ.get("GEMINI_API_KEY")
# _translation_client = genai.Client(api_key=_translation_key)

# # ---------------------------------------------------------------------------
# # Language coverage: map Indian states/UTs -> a language key.
# # Extend this as you onboard more regions; unmapped states fall back to Hindi,
# # which is the most broadly understood option across most of India.
# # ---------------------------------------------------------------------------
# STATE_TO_LANGUAGE = {
#     "odisha": "odia",
#     "west bengal": "bengali",
#     "tamil nadu": "tamil",
#     "andhra pradesh": "telugu",
#     "telangana": "telugu",
#     "maharashtra": "marathi",
#     "gujarat": "gujarati",
#     "karnataka": "kannada",
#     "kerala": "malayalam",
#     "punjab": "punjabi",
#     "uttar pradesh": "hindi",
#     "bihar": "hindi",
#     "madhya pradesh": "hindi",
#     "rajasthan": "hindi",
#     "haryana": "hindi",
#     "delhi": "hindi",
#     "jharkhand": "hindi",
#     "chhattisgarh": "hindi",
#     "uttarakhand": "hindi",
#     "himachal pradesh": "hindi",
# }

# # Per-language voice/engine configuration.
# # gtts_lang = None means gTTS has no usable voice for that language, so the
# # fallback path will render a generic spoken notice instead of silent-failing.
# LANGUAGE_MAP = {
#     "odia":       {"code": "or", "voice": "or-IN-SubhasiniNeural", "gtts_lang": None},
#     "hindi":      {"code": "hi", "voice": "hi-IN-SwaraNeural",     "gtts_lang": "hi"},
#     "bengali":    {"code": "bn", "voice": "bn-IN-TanishaaNeural",  "gtts_lang": "bn"},
#     "tamil":      {"code": "ta", "voice": "ta-IN-PallaviNeural",   "gtts_lang": "ta"},
#     "telugu":     {"code": "te", "voice": "te-IN-ShrutiNeural",    "gtts_lang": "te"},
#     "marathi":    {"code": "mr", "voice": "mr-IN-AarohiNeural",    "gtts_lang": "mr"},
#     "gujarati":   {"code": "gu", "voice": "gu-IN-DhwaniNeural",    "gtts_lang": "gu"},
#     "kannada":    {"code": "kn", "voice": "kn-IN-SapnaNeural",     "gtts_lang": "kn"},
#     "malayalam":  {"code": "ml", "voice": "ml-IN-SobhanaNeural",   "gtts_lang": "ml"},
#     "punjabi":    {"code": "pa", "voice": "hi-IN-SwaraNeural",     "gtts_lang": "pa"},  # edge-tts has no pa-IN voice; falls back to Hindi voice for TTS, gTTS still uses Punjabi
# }

# # Short spoken notice used only if cleaned_text ends up empty, per language.
# # Falls back to Hindi if a language isn't listed here.
# FALLBACK_MESSAGES = {
#     "odia": "ଆପଣଙ୍କ ଫସଲର ରିପୋର୍ଟ ପ୍ରସ୍ତୁତ ହୋଇଯାଇଛି।",
#     "hindi": "आपकी फसल की रिपोर्ट तैयार हो गई है।",
#     "bengali": "আপনার ফসলের রিপোর্ট প্রস্তুত হয়ে গেছে।",
#     "tamil": "உங்கள் பயிர் அறிக்கை தயாராகிவிட்டது.",
#     "telugu": "మీ పంట నివేదిక సిద్ధమైంది.",
#     "marathi": "तुमचा पीक अहवाल तयार झाला आहे.",
#     "gujarati": "તમારો પાક અહેવાલ તૈયાર થઈ ગયો છે.",
#     "kannada": "ನಿಮ್ಮ ಬೆಳೆ ವರದಿ ಸಿದ್ಧವಾಗಿದೆ.",
#     "malayalam": "നിങ്ങളുടെ വിള റിപ്പോർട്ട് തയ്യാറായി.",
#     "punjabi": "ਤੁਹਾਡੀ ਫਸਲ ਦੀ ਰਿਪੋਰਟ ਤਿਆਰ ਹੋ ਗਈ ਹੈ।",
# }


# def _resolve_language_key(state_context: str) -> str:
#     """Map a free-text state name to a language key, defaulting to Hindi."""
#     normalized = state_context.lower().strip()
#     return STATE_TO_LANGUAGE.get(normalized, "hindi")


# def translate_treatment_plan(treatment_plan, state_context: str) -> tuple[str, str]:
#     """
#     Localizes the treatment plan into the primary language spoken in
#     state_context, returning (display_text, speech_text).

#     display_text: WhatsApp-formatted (single *asterisk* bold), symbols like
#     32°C / ₹500 / 5kg left as normal written form since it's read visually.

#     speech_text: fully spelled out in words (units, currency, temperature)
#     in that language's natural spoken form, no markdown symbols — this is
#     what TTS will read, so nothing here should rely on post-hoc regex swaps.
#     """
#     prompt = f"""
# Translate this agricultural treatment plan into the primary language spoken in
# {state_context}, India (e.g. Odia for Odisha, Tamil for Tamil Nadu, Marathi for
# Maharashtra, etc. — infer the correct language from the state name).

# Return JSON with two keys:
# - "display_text": for WhatsApp chat. Use WhatsApp formatting only (single
#   *asterisk* for bold). Numbers/units/currency can stay in normal written form
#   (32°C, ₹500, 5kg) since this is read visually.
# - "speech_text": for text-to-speech narration in the same language. No markdown
#   symbols at all. Every number, unit, currency amount, and temperature must be
#   fully spelled out as words in that language, following that language's natural
#   spoken conventions (grammar, gender agreement, etc.), since a TTS engine reads
#   this literally and cannot pronounce symbols correctly.

# Treatment plan:
# {treatment_plan}
# """
#     response = _translation_client.models.generate_content(
#         model="gemini-3.5-flash",
#         contents=[prompt],
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#             temperature=0.1
#         )
#     )
#     result = json.loads(response.text)
#     return result["display_text"], result["speech_text"]


# async def generate_voice_note(text: str, output_filename: str, state_context: str = "odisha") -> str:
#     """
#     Converts text to an .mp3 voice note using Edge-TTS, falling back gracefully
#     to gTTS (or a short spoken notice) to prevent pipeline failure.

#     `text` is expected to already be the LLM-produced speech_text — i.e. units,
#     currency, and temperature already spelled out in words for the target
#     language. This function only strips residual markdown/punctuation symbols;
#     it does not perform any language-specific word substitution.
#     """
#     lang_key = _resolve_language_key(state_context)
#     config = LANGUAGE_MAP.get(lang_key, LANGUAGE_MAP["hindi"])
#     voice_profile = config["voice"]
#     gtts_lang = config["gtts_lang"]

#     os.makedirs("static", exist_ok=True)
#     file_path = os.path.join("static", output_filename)

#     # Language-agnostic punctuation/symbol scrubbing only.
#     # (No hardcoded word substitutions here — speech_text already arrives
#     # pre-localized per language from translate_treatment_plan.)
#     cleaned_text = text
#     cleaned_text = cleaned_text.replace("|", " ")
#     cleaned_text = cleaned_text.replace("।", " . ")  # traditional danda -> spoken pause
#     cleaned_text = re.sub(r"#{1,6}\s*", "", cleaned_text)
#     cleaned_text = re.sub(r"-\s*-\s*-\s*", "", cleaned_text)
#     cleaned_text = re.sub(r"[\*#_`\-─│┌┐└┘├┤┬┴┼:!,?]", " ", cleaned_text)
#     cleaned_text = cleaned_text.replace("[", "").replace("]", "")

#     cleaned_text = re.sub(r"\n+", " . \n", cleaned_text)
#     cleaned_text = re.sub(r" +", " ", cleaned_text).strip()

#     if not cleaned_text:
#         cleaned_text = FALLBACK_MESSAGES.get(lang_key, FALLBACK_MESSAGES["hindi"])

#     # Strategy A: Edge TTS (primary, higher quality, widest voice coverage)
#     try:
#         logger.info(f"Attempting Edge TTS generation via voice target: {voice_profile}...")
#         communicate = edge_tts.Communicate(cleaned_text, voice_profile)
#         await communicate.save(file_path)

#         if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#             logger.info(f"Edge TTS compilation successful and verified: {file_path}")
#             return file_path
#         else:
#             raise Exception("Edge TTS generated an empty payload file structure.")

#     except Exception as edge_err:
#         logger.warning(f"Primary Edge TTS dropped out ({str(edge_err)}). Activating emergency fallback...")

#         # Strategy B: gTTS fallback, or a short generic spoken notice if the
#         # language isn't supported by gTTS at all.
#         try:
#             if gtts_lang:
#                 logger.info("Compiling speech file via localized gTTS framework...")
#                 tts = gTTS(text=cleaned_text, lang=gtts_lang, slow=False)
#                 tts.save(file_path)
#             else:
#                 logger.info("Target language lacks native gTTS support. Rendering safety notification audio asset...")
#                 safety_text = "Your crop diagnosis report has been successfully generated. Please check your text messages."
#                 tts = gTTS(text=safety_text, lang="en", slow=False)
#                 tts.save(file_path)

#             if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#                 logger.info(f"Emergency safety audio asset compile complete: {file_path}")
#                 return file_path
#             else:
#                 raise RuntimeError("Backup TTS system returned a dead size metrics envelope.")
#         except Exception as fallback_err:
#             logger.error(f"Critical Layer Exhaustion: {str(fallback_err)}")
#             raise fallback_err