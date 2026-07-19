import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, Form, BackgroundTasks, Response, Request
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import re
from fastapi.staticfiles import StaticFiles
from accessibilitylayer import translate_treatment_plan, generate_voice_note
from twilio.rest import Client

from enrichmentlayer import fetch_weather_metrics, fetch_mandi_prices
from agentlayer import run_agentic_reasoning

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Crop Diagnosis Agent AI Gateway")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the client specifically with the dedicated Vision API Key
vision_key = os.environ.get("GEMINI_API_KEY")
vision_client = genai.Client(api_key=vision_key)


class CropAnalysisSchema(BaseModel):
    is_valid_crop_image: bool = Field(description="True if the image contains an identifiable agricultural plant or leaf context.")
    crop_name: str = Field(description="Common name of the crop identified, or 'Unknown'.")
    disease_detected: str = Field(description="Name of the pathology found, or 'Healthy' if none.")
    confidence_score: float = Field(description="Confidence rating of the identification between 0.0 and 1.0.")
    visual_symptoms: list[str] = Field(description="List of structural visual anomalies like spots, wilting, or lesions.")
    search_query_key: str = Field(description="Simplified hyphenated keyword string for cataloging.")


def clean_for_whatsapp(text: str) -> str:
    """Convert standard Markdown into WhatsApp's own formatting syntax."""
    # Bold: **text** -> *text*  (do this BEFORE touching single asterisks)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Headers: #### Heading -> just "Heading" on its own line
    text = re.sub(r'^#+\s*(.*)', r'\1', text, flags=re.MULTILINE)
    # Horizontal rules: --- -> drop entirely
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse extra blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_for_tts(text: str) -> str:
    """
    Strip markdown symbols so TTS reads clean, natural sentences.
    NOTE: Unit/currency/temperature spelling-out (e.g. 32°C -> "thirty-two
    degrees Celsius" in the target language) is handled upstream inside
    translate_treatment_plan(), per language, since that generalizes across
    every Indian language instead of a hardcoded regex dictionary here.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = text.replace('*', '').replace('#', '')
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def chunk_message_text(text: str, max_chars: int = 1500) -> list[str]:
    """Splits text into chunks under the max_chars limit without cutting off lines
    where possible. Guards against emitting empty chunks (which Twilio rejects
    with error 21619) and hard-splits any single line that's longer than
    max_chars on its own."""
    chunks = []
    current_chunk = ""
 
    for line in text.splitlines(keepends=True):
        if current_chunk and len(current_chunk) + len(line) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += line
 
        # A single line can itself exceed max_chars (e.g. one long unbroken
        # paragraph in the translated text). Hard-split it rather than
        # letting current_chunk grow past the limit.
        while len(current_chunk) > max_chars:
            chunks.append(current_chunk[:max_chars].strip())
            current_chunk = current_chunk[max_chars:]
 
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
 
    # Defensive final filter: never return an empty chunk.
    return [c for c in chunks if c.strip()]
 


async def analyze_crop_image(media_url: str) -> CropAnalysisSchema:
    """
    Leverages Gemini via the vision-specific key to process incoming media URLs.
    """
    logger.info(f"Passing asset to Gemini Vision parsing engine: {media_url}")

    prompt = "Analyze this agricultural field image. Extract the explicit health state, commodity profile, and abnormalities."

    response = vision_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_uri(file_uri=media_url, mime_type="image/jpeg"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CropAnalysisSchema,
            temperature=0.1
        )
    )

    return CropAnalysisSchema.model_validate_json(response.text)


async def orchestrator_background_worker(
    farmer_phone: str,
    media_url: str,
    lat: Optional[str],
    lon: Optional[str],
    base_url: str
):
    """
    Asynchronous internal task pipeline. Concurrently manages vision processing,
    cached tool data lookups, and agentic reasoning layers.
    """
    try:
        logger.info(f"Background worker kicked off for client: {farmer_phone}")

        local_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        local_token = os.environ.get("TWILIO_AUTH_TOKEN")
        local_sender = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

        active_twilio_client = Client(local_sid, local_token)

        # 1. Execute Multimodal Structured Vision Extraction
        analysis = await analyze_crop_image(media_url)
        logger.info(f"Structured analysis successfully compiled: {analysis.model_dump_json()}")

        if not analysis.is_valid_crop_image:
            logger.warning("Invalid field image context provided by farmer. Terminating execution loop.")
            return

        # 2. Concurrent Data Enrichment
        logger.info(f"Launching concurrent environmental enrichment pipelines for: {analysis.crop_name}")

        weather_data, market_data = await asyncio.gather(
            fetch_weather_metrics(lat, lon),
            fetch_mandi_prices(analysis.crop_name, lat, lon)
        )

        logger.info(f"Context Gathering Complete! Weather: {weather_data} | Mandi: {market_data}")

        logger.info("Engaging agentic reasoning layer...")
        treatment_plan = run_agentic_reasoning(analysis, weather_data, market_data)

        # --- STEP 4. ACCESSIBILITY LAYER INTEGRATION ---
        # farmer_state is the geocoded location of the farmer (source of truth
        # for language). market_data["state"] can differ if mandi price data
        # came from a nationwide fallback search, so we never use that for
        # language selection.
        detected_state = market_data.get("farmer_state") or "odisha"

        # A. Translate the treatment plan into the farmer's regional language.
        # translate_treatment_plan() returns a (display_text, speech_text) tuple:
        #   - display_text: WhatsApp-formatted, symbols like °C/₹/kg left as-is
        #   - speech_text: fully spelled out (units, currency, temperature) in
        #     that language's natural spoken form, no markdown symbols
        raw_display_text, raw_speech_text = translate_treatment_plan(
            treatment_plan, state_context=detected_state
        )

        whatsapp_text = clean_for_whatsapp(raw_display_text)
        tts_text = clean_for_tts(raw_speech_text)

        logger.info(f"Local Language Text Ready for Dispatch:\n{whatsapp_text}")

        # B. Dispatch the localized text message, chunked to respect Twilio's
        # 1600-char limit (Error 21617 otherwise).
        logger.info("Splitting text report to respect Twilio character boundaries...")
        text_chunks = chunk_message_text(whatsapp_text, max_chars=1500)

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Dispatching text report chunk {i+1}/{len(text_chunks)} via Twilio...")
            active_twilio_client.messages.create(
                body=chunk,
                from_=local_sender,
                to=farmer_phone
            )
            # A tiny sleep ensures the messages arrive in chronological order on WhatsApp
            await asyncio.sleep(0.5)

        logger.info("All text report chunks successfully pushed to farmer chat.")

        # C. Generate Natural Audio Note File (TTS-safe text, not the display text).
        # Returns None if no working TTS voice exists for this language on
        # either engine (e.g. Odia right now) — in that case we skip audio
        # entirely rather than sending a misleading placeholder.
        safe_phone_string = "".join(filter(str.isdigit, farmer_phone))
        audio_filename = f"report_{safe_phone_string}.mp3"

        voice_result = await generate_voice_note(tts_text, audio_filename, state_context=detected_state)

        if voice_result is None:
            logger.info("No TTS voice available for this language — skipping audio dispatch, text report stands alone.")
        else:
            # D. Assemble the public static audio asset link
            public_audio_url = f"{base_url}/static/{audio_filename}"
            logger.info(f"Public Audio Note Asset Endpoint Ready: {public_audio_url}")

            # E. Dispatch the companion audio voice note right after the text
            logger.info("Dispatching companion audio voice note via Twilio WhatsApp Gateway...")
            msg = active_twilio_client.messages.create(
                from_=local_sender,
                media_url=[public_audio_url],
                to=farmer_phone
            )
            logger.info(f"Message SID: {msg.sid}, initial status: {msg.status}")

            # Twilio processes media asynchronously — wait a moment, then re-check
            # the real delivery status rather than trusting the 201 alone.
            await asyncio.sleep(3)
            updated = active_twilio_client.messages(msg.sid).fetch()
            logger.info(f"Final status: {updated.status}, error_code: {updated.error_code}, error_message: {updated.error_message}")

    except Exception as e:
        logger.error(f"Critical execution thread failure within worker loop context: {str(e)}")


@app.post("/webhook")
async def incoming_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    MediaUrl0: str = Form(...),
    Body: Optional[str] = Form(None),
    Latitude: Optional[str] = Form(None),
    Longitude: Optional[str] = Form(None)
):
    logger.info(f"Ingress payload intercepted from sender gateway: {From}")

    # Dynamically extract the root domain string (handles ngrok or production domains automatically)
    extracted_base_url = str(request.base_url).rstrip('/')

    # If coordinates are missing from the fields, parse the WhatsApp caption string
    final_lat = Latitude
    final_lon = Longitude

    if not final_lat or not final_lon:
        if Body:
            logger.info(f"Checking caption body for coordinates: '{Body}'")
            lat_match = re.search(r"lat:\s*([\d\.]+)", Body, re.IGNORECASE)
            lon_match = re.search(r"lon:\s*([\d\.]+)", Body, re.IGNORECASE)

            if lat_match and lon_match:
                final_lat = lat_match.group(1)
                final_lon = lon_match.group(1)
                logger.info(f"Successfully extracted coordinates from caption text -> Lat: {final_lat}, Lon: {final_lon}")

    background_tasks.add_task(
        orchestrator_background_worker,
        farmer_phone=From,
        media_url=MediaUrl0,
        lat=final_lat,
        lon=final_lon,
        base_url=extracted_base_url
    )

    twiml_response = (
        "<Response>"
        "<Message>Received your image! Our AI Agronomist is analyzing your field samples, "
        "checking local weather trends, and compiling market reports now...</Message>"
        "</Response>"
    )
    return Response(content=twiml_response, media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# import os
# import asyncio
# import logging
# from typing import Optional
# from fastapi import FastAPI, Form, BackgroundTasks, Response
# from pydantic import BaseModel, Field
# from google import genai
# from fastapi import Request
# from google.genai import types
# import re
# from fastapi.staticfiles import StaticFiles
# from accessibilitylayer import translate_treatment_plan, generate_voice_note
# from twilio.rest import Client  # <-- Add this import

# app = FastAPI(title="Crop Diagnosis Agent AI Gateway")

# # Initialize your outbound Twilio Gateway Client

# # Import our async caching enrichment and agent layers
# from enrichmentlayer import fetch_weather_metrics, fetch_mandi_prices
# from agentlayer import run_agentic_reasoning
# TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
# TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")

# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("main")

# app = FastAPI(title="Crop Diagnosis Agent AI Gateway")
# os.makedirs("static", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# # Initialize the client specifically with the dedicated Vision API Key
# vision_key = os.environ.get("GEMINI_API_KEY")
# vision_client = genai.Client(api_key=vision_key)

# class CropAnalysisSchema(BaseModel):
#     is_valid_crop_image: bool = Field(description="True if the image contains an identifiable agricultural plant or leaf context.")
#     crop_name: str = Field(description="Common name of the crop identified, or 'Unknown'.")
#     disease_detected: str = Field(description="Name of the pathology found, or 'Healthy' if none.")
#     confidence_score: float = Field(description="Confidence rating of the identification between 0.0 and 1.0.")
#     visual_symptoms: list[str] = Field(description="List of structural visual anomalies like spots, wilting, or lesions.")
#     search_query_key: str = Field(description="Simplified hyphenated keyword string for cataloging.")

# def clean_for_whatsapp(text: str) -> str:
#     """Convert standard Markdown into WhatsApp's formatting syntax."""
#     # Bold: **text** -> *text*  (do this BEFORE touching single asterisks)
#     text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
#     # Headers: #### Heading -> just "Heading" on its own line
#     text = re.sub(r'^#+\s*(.*)', r'\1', text, flags=re.MULTILINE)
#     # Horizontal rules: --- -> drop entirely
#     text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
#     # Collapse extra blank lines left behind
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# def normalize_units_for_tts(text: str) -> str:
#     """Expand symbols/abbreviations into full Odia words so TTS pronounces them correctly."""
#     replacements = [
#         (r'(\d+)\s*°\s*C', r'\1 ଡିଗ୍ରୀ ସେଲସିୟସ'),      # 32°C -> 32 ଡିଗ୍ରୀ ସେଲସିୟସ
#         (r'₹\s?(\d+)', r'\1 ଟଙ୍କା'),                    # ₹500 -> 500 ଟଙ୍କା
#         (r'\bINR\s?(\d+)', r'\1 ଟଙ୍କା'),                # INR 500 -> 500 ଟଙ୍କା
#         (r'(\d+)\s*kg\b', r'\1 କିଲୋଗ୍ରାମ'),
#         (r'(\d+)\s*g\b', r'\1 ଗ୍ରାମ'),
#         (r'(\d+)\s*ml\b', r'\1 ମିଲିଲିଟର'),
#         (r'(\d+)\s*l\b', r'\1 ଲିଟର', ),
#         (r'(\d+)\s*ha\b', r'\1 ଏକର'),
#         (r'(\d+)\s*%', r'\1 ପ୍ରତିଶତ'),
#     ]
#     for pattern, repl in replacements:
#         text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
#     return text

# def clean_for_tts(text: str) -> str:
#     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
#     text = re.sub(r'\*(.*?)\*', r'\1', text)
#     text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
#     text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
#     text = text.replace('*', '').replace('#', '')
#     text = normalize_units_for_tts(text)          # <-- new step, run BEFORE collapsing whitespace
#     text = re.sub(r'\n{2,}', '. ', text)
#     text = re.sub(r'\s{2,}', ' ', text)
#     return text.strip()

# def chunk_message_text(text: str, max_chars: int = 1500) -> list[str]:
#     """Splits text into chunks under the max_chars limit without cutting off lines."""
#     chunks = []
#     current_chunk = ""
    
#     for line in text.splitlines(keepends=True):
#         if len(current_chunk) + len(line) > max_chars:
#             chunks.append(current_chunk.strip())
#             current_chunk = line
#         else:
#             current_chunk += line
            
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# async def analyze_crop_image(media_url: str) -> CropAnalysisSchema:
#     """
#     Leverages Gemini 2.5 Flash via the vision-specific key to process incoming media URLs.
#     """
#     logger.info(f"Passing asset to Gemini Vision parsing engine: {media_url}")
    
#     prompt = "Analyze this agricultural field image. Extract the explicit health state, commodity profile, and abnormalities."
    
#     response = vision_client.models.generate_content(
#         model="gemini-3.5-flash",
#         contents=[
#             types.Part.from_uri(file_uri=media_url, mime_type="image/jpeg"),
#             prompt
#         ],
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=CropAnalysisSchema,
#             temperature=0.1
#         )
#     )
    
#     return CropAnalysisSchema.model_validate_json(response.text)

# async def orchestrator_background_worker(
#     farmer_phone: str, 
#     media_url: str, 
#     lat: Optional[str], 
#     lon: Optional[str],
#     base_url: str
# ):
#     """
#     Asynchronous internal task pipeline. Concurrently manages vision processing,
#     cached tool data lookups, and LangGraph cognitive graph layers.
#     """
#     try:
#         logger.info(f"Background worker kicked off for client: {farmer_phone}")
        
#         # --- NEW: INTERNALIZED TWILIO SCOPE INITIALIZATION ---
#         # Resolving environment strings locally inside the thread worker context
#         local_sid = os.environ.get("TWILIO_ACCOUNT_SID")
#         local_token = os.environ.get("TWILIO_AUTH_TOKEN")
#         local_sender = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        
#         # Direct local client instantiation matching test_twilio.py environment condition
#         active_twilio_client = Client(local_sid, local_token)
        
#         # 1. Execute Multimodal Structured Vision Extraction
#         analysis = await analyze_crop_image(media_url)
#         logger.info(f"Structured analysis successfully compiled: {analysis.model_dump_json()}")

#         if not analysis.is_valid_crop_image:
#             logger.warning("Invalid field image context provided by farmer. Terminating execution loop.")
#             return

#         # 2. Concurrent Data Enrichment via Async Redis Caching Layer
#         logger.info(f"Launching concurrent environmental enrichment pipelines for: {analysis.crop_name}")
        
#         weather_data, market_data = await asyncio.gather(
#             fetch_weather_metrics(lat, lon),
#             fetch_mandi_prices(analysis.crop_name, lat, lon)
#         )
        
#         logger.info(f"Context Gathering Complete! Weather: {weather_data} | Mandi: {market_data}")

#         logger.info("Engaging LangGraph cognitive reasoning graph layers...")
#         treatment_plan = run_agentic_reasoning(analysis, weather_data, market_data)
        
#         # --- STEP 4. ACCESSIBILITY LAYER INTEGRATION ---
#         # Extract state context from market payload to determine localized speech language (default to Odisha)
#         detected_state = market_data.get("state", "odisha")
        
#         # A. Translate Text Plan into Odia/Regional Language
#         local_treatment_text = translate_treatment_plan(treatment_plan, state_context=detected_state)
#         logger.info(f"Local Language Text Ready for Dispatch:\n{local_treatment_text}")
        
#         # B. Dispatch Option 1: Send the clean Localized Text Message (Chunked to prevent Error 21617)
#         logger.info("Splitting text report to respect Twilio character boundaries...")
#         text_chunks = chunk_message_text(local_treatment_text, max_chars=1500)
        
#         for i, chunk in enumerate(text_chunks):
#             logger.info(f"Dispatching text report chunk {i+1}/{len(text_chunks)} via Twilio...")
#             active_twilio_client.messages.create(
#                 body=chunk,
#                 from_=local_sender,
#                 to=farmer_phone
#             )
#             # A tiny sleep ensures the messages arrive in chronological order on WhatsApp
#             await asyncio.sleep(0.5) 
            
#         logger.info("All text report chunks successfully pushed to farmer chat.")

#         # C. Generate Natural Audio Note File
#         safe_phone_string = "".join(filter(str.isdigit, farmer_phone))
#         audio_filename = f"report_{safe_phone_string}.mp3"
        
#         # We will attempt the deep-cleaned Edge-TTS execution
#         # (If it drops out, our fallback generates the safety audio file)
#         await generate_voice_note(local_treatment_text, audio_filename, state_context=detected_state)
        
#         # D. Assemble the public static audio asset link
#         public_audio_url = f"{base_url}/static/{audio_filename}"
#         logger.info(f"Public Audio Note Asset Endpoint Ready: {public_audio_url}")

#         # E. Dispatch Option 3: Send the Audio Voice Note right after the text
#         logger.info("Dispatching companion audio voice note via Twilio WhatsApp Gateway...")
#         active_twilio_client.messages.create(
#             from_=local_sender,
#             media_url=[public_audio_url],
#             to=farmer_phone
#         )
#         logger.info("Outbound voice payload successfully pushed to farmer chat.")

#     except Exception as e:
#         logger.error(f"Critical execution thread failure within worker loop context: {str(e)}")
# @app.post("/webhook")
# async def incoming_whatsapp_webhook(
#     request: Request,                         # <-- 1. Inject the ASGI Request object here
#     background_tasks: BackgroundTasks,
#     From: str = Form(...),
#     MediaUrl0: str = Form(...),
#     Body: Optional[str] = Form(None),         
#     Latitude: Optional[str] = Form(None),
#     Longitude: Optional[str] = Form(None)
# ):
#     logger.info(f"Ingress payload intercepted from sender gateway: {From}")
    
#     # Dynamically extract the root domain string (handles ngrok or production domains automatically)
#     extracted_base_url = str(request.base_url).rstrip('/')
    
#     # 2. If coordinates are missing from the fields, parse the WhatsApp caption string
#     final_lat = Latitude
#     final_lon = Longitude

#     if not final_lat or not final_lon:
#         if Body:
#             logger.info(f"Checking caption body for coordinates: '{Body}'")
#             # Regex to match numeric values after lat: and lon: labels
#             lat_match = re.search(r"lat:\s*([\d\.]+)", Body, re.IGNORECASE)
#             lon_match = re.search(r"lon:\s*([\d\.]+)", Body, re.IGNORECASE)
            
#             if lat_match and lon_match:
#                 final_lat = lat_match.group(1)
#                 final_lon = lon_match.group(1)
#                 logger.info(f"Successfully extracted coordinates from caption text -> Lat: {final_lat}, Lon: {final_lon}")

#     # 3. Pass all matching explicit parameters to the background worker task structure
#     background_tasks.add_task(
#         orchestrator_background_worker,
#         farmer_phone=From,
#         media_url=MediaUrl0,
#         lat=final_lat,
#         lon=final_lon,
#         base_url=extracted_base_url          # <-- 2. Safely supply the missing positional payload link here!
#     )
    
#     twiml_response = (
#         "<Response>"
#         "<Message>Received your image! Our AI Agronomist is analyzing your field samples, "
#         "checking local weather trends, and compiling market reports now...</Message>"
#         "</Response>"
#     )
#     return Response(content=twiml_response, media_type="application/xml")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)