# import asyncio  
# from enrichmentlayer import fetch_weather_metrics, fetch_mandi_prices
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from fastapi import FastAPI, Form, BackgroundTasks, Response
# from typing import Optional
# import logging
# from visionlayer import analyze_crop_image, PlantDiagnostic

# # Setup structural basic logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Croplens Ag-Agent Gateway")

# async def orchestrator_background_worker(farmer_phone: str, media_url: str, lat: Optional[str], lon: Optional[str]):
#     """
#     This background worker functions completely detached from the HTTP response thread,
#     giving you unlimited time to execute Vision APIs, Context Tool Fetches, and TTS generation.
#     """
#     try:
#         logger.info(f"Background worker kicked off for client: {farmer_phone}")
        
#         # --- Milestone 2: Execute Structured Vision Layer ---
#         analysis: PlantDiagnostic = await analyze_crop_image(media_url)
#         logger.info(f"Structured analysis successfully compiled: {analysis}")

#         if not analysis.is_valid_crop_image:
#             # TODO: Fire immediate Twilio SMS back: "Please upload a clear leaf view"
#             logger.warning("Invalid field image context provided by farmer.")
#             return

#         # --- Milestone 3: Tool Integration Layer Hook (Placeholder) ---
#         # coordinates = {"lat": lat, "lon": lon}
#         # weather = await fetch_weather_metrics(lat, lon)
#         # market = await fetch_mandi_prices(analysis.crop_name, lat, lon)
        
#         # --- Milestone 4 & 5: LangGraph Agent Engine & TTS Delivery Hook (Placeholder) ---
#         # final_plan_text = run_agentic_graph(analysis, weather, market)
#         # audio_url = generate_tts_voice_note(final_plan_text)
#         # send_whatsapp_voice_note(farmer_phone, audio_url)
        
#         logger.info(f"Successfully processed loop execution context for {farmer_phone}")

#     except Exception as e:
#         logger.error(f"Critical operational error occurred within background loop execution thread: {str(e)}")

# @app.post("/webhook")
# async def incoming_whatsapp_webhook(
#     background_tasks: BackgroundTasks,
#     From: str = Form(...),                  # Farmer phone identifier number
#     MediaUrl0: Optional[str] = Form(None),   # Hosted image web pathway path
#     Latitude: Optional[str] = Form(None),   # Geolocation latitude string
#     Longitude: Optional[str] = Form(None)   # Geolocation longitude string
# ):
#     """
#     Ingestion hub endpoint targeting incoming Twilio standard Form payload mappings.
#     Forces non-blocking handoffs to guarantee sub-second execution timelines.
#     """
#     logger.info(f"Incoming transmission capture triggered by phone number node: {From}")

#     # Validate image element presence
#     if not MediaUrl0:
#         # Twilio expects a valid TwiML XML response structure back if returning text directly
#         twiml_fallback = (
#             "<Response><Message>Welcome to Croplens! Please send a clear photograph "
#             "of the damaged plant leaf along with your shared location pin to begin diagnosis.</Message></Response>"
#         )
#         return Response(content=twiml_fallback, media_type="application/xml")

#     # Hand off heavy operational executions to the worker thread safely
#     background_tasks.add_task(
#         orchestrator_background_worker,
#         farmer_phone=From,
#         media_url=MediaUrl0,
#         lat=Latitude,
#         lon=Longitude
#     )

#     # Return immediate acknowledgments back to the platform gateway under 100 milliseconds
#     twiml_processing = (
#         "<Response><Message>Image receipt acknowledged. Analyzing leaf patterns "
#         "and querying local mandi rates now. Please wait for the voice note transmission...</Message></Response>"
#     )
#     return Response(content=twiml_processing, media_type="application/xml")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# async def orchestrator_background_worker(farmer_phone: str, media_url: str, lat: Optional[str], lon: Optional[str]):
#     try:
#         logger.info(f"Background worker kicked off for client: {farmer_phone}")
        
#         # --- Milestone 2: Execute Structured Vision Layer ---
#         analysis = await analyze_crop_image(media_url)
#         logger.info(f"Structured analysis successfully compiled: {analysis}")

#         if not analysis.is_valid_crop_image:
#             logger.warning("Invalid field image context provided by farmer.")
#             return

#         # --- Milestone 3: Concurrent Data Enrichment ---
#         logger.info(f"Launching concurrent environmental enrichment threads for crop: {analysis.crop_name}")
        
#         # Run both API tasks simultaneously
#         weather_data, market_data = await asyncio.gather(
#             fetch_weather_metrics(lat, lon),
#             fetch_mandi_prices(analysis.crop_name, lat, lon)
#         )
        
#         logger.info(f"Enrichment Complete! Weather context: {weather_data}")
#         logger.info(f"Enrichment Complete! Mandi context: {market_data}")

#         # --- Upcoming Milestones 4 & 5 ---
#         # Next, we pass `analysis`, `weather_data`, and `market_data` into our multi-agent brain graph!
        
#     except Exception as e:
#         logger.error(f"Critical operational error occurred within background loop execution thread: {str(e)}")
import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, Form, BackgroundTasks, Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from fastapi.staticfiles import StaticFiles
from accessibilitylayer import translate_treatment_plan, generate_voice_note

# Import our async caching enrichment and agent layers
from enrichmentlayer import fetch_weather_metrics, fetch_mandi_prices
from agentlayer import run_agentic_reasoning

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

async def analyze_crop_image(media_url: str) -> CropAnalysisSchema:
    """
    Leverages Gemini 2.5 Flash via the vision-specific key to process incoming media URLs.
    """
    logger.info(f"Passing asset to Gemini Vision parsing engine: {media_url}")
    
    prompt = "Analyze this agricultural field image. Extract the explicit health state, commodity profile, and abnormalities."
    
    response = vision_client.models.generate_content(
        model="gemini-2.5-flash",
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

async def orchestrator_background_worker(farmer_phone: str, media_url: str, lat: Optional[str], lon: Optional[str]):
    """
    Asynchronous internal task pipeline. Concurrently manages vision processing,
    cached tool data lookups, and LangGraph cognitive graph layers.
    """
    try:
        logger.info(f"Background worker kicked off for client: {farmer_phone}")
        
        # 1. Execute Multimodal Structured Vision Extraction
        analysis = await analyze_crop_image(media_url)
        logger.info(f"Structured analysis successfully compiled: {analysis.model_dump_json()}")

        if not analysis.is_valid_crop_image:
            logger.warning("Invalid field image context provided by farmer. Terminating execution loop.")
            return

        # 2. Concurrent Data Enrichment via Async Redis Caching Layer
        logger.info(f"Launching concurrent environmental enrichment pipelines for: {analysis.crop_name}")
        
        weather_data, market_data = await asyncio.gather(
            fetch_weather_metrics(lat, lon),
            fetch_mandi_prices(analysis.crop_name, lat, lon)
        )
        
        logger.info(f"Context Gathering Complete! Weather: {weather_data} | Mandi: {market_data}")

        logger.info("Engaging LangGraph cognitive reasoning graph layers...")
        treatment_plan = run_agentic_reasoning(analysis, weather_data, market_data)
        
        # --- NEW: STEP 4. ACCESSIBILITY LAYER INTEGRATION ---
        # Extract state context from market payload to determine localized speech language (default to Odisha)
        detected_state = market_data.get("state", "odisha")
        
        # A. Translate Text Plan
        local_treatment_text = translate_treatment_plan(treatment_plan, state_context=detected_state)
        logger.info(f"Translated Text Outcome:\n{local_treatment_text}")
        
        # B. Generate Natural Audio Note File
        # Strip out "whatsapp:", "+", and spaces to ensure a clean numeric-only filename
        safe_phone_string = "".join(filter(str.isdigit, farmer_phone))
        audio_filename = f"report_{safe_phone_string}.mp3"
        
        await generate_voice_note(local_treatment_text, audio_filename, state_context=detected_state)
        
        # C. Construct the public link that Twilio can read over ngrok
        # Replace this with a dynamic variable or your hardcoded active ngrok base address
        NGROK_BASE_URL = "https://sterile-deviate-worry.ngrok-free.dev"  
        public_audio_url = f"{NGROK_BASE_URL}/static/{audio_filename}"
        logger.info(f"Public Audio Note Asset Endpoint Ready: {public_audio_url}")

        # D. Trigger Twilio Outbound Dispatch Action
        # (This is where you execute the Twilio client.messages.create call, 
        # passing media_url=[public_audio_url] back to the sender)
        logger.info(f"Pushed voice dispatch packet back to farmer destination channel: {farmer_phone}")

    except Exception as e:
        logger.error(f"Critical execution thread failure within worker loop context: {str(e)}")

import re

@app.post("/webhook")
async def incoming_whatsapp_webhook(
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    MediaUrl0: str = Form(...),
    Body: Optional[str] = Form(None),         # <-- 1. Catch the text caption here
    Latitude: Optional[str] = Form(None),
    Longitude: Optional[str] = Form(None)
):
    logger.info(f"Ingress payload intercepted from sender gateway: {From}")
    
    # 2. If coordinates are missing from the fields, parse the WhatsApp caption string
    final_lat = Latitude
    final_lon = Longitude

    if not final_lat or not final_lon:
        if Body:
            logger.info(f"Checking caption body for coordinates: '{Body}'")
            # Regex to match numeric values after lat: and lon: labels
            lat_match = re.search(r"lat:\s*([\d\.]+)", Body, re.IGNORECASE)
            lon_match = re.search(r"lon:\s*([\d\.]+)", Body, re.IGNORECASE)
            
            if lat_match and lon_match:
                final_lat = lat_match.group(1)
                final_lon = lon_match.group(1)
                logger.info(f"Successfully extracted coordinates from caption text -> Lat: {final_lat}, Lon: {final_lon}")

    # 3. Pass the parsed coordinates to the background execution loop
    background_tasks.add_task(
        orchestrator_background_worker,
        farmer_phone=From,
        media_url=MediaUrl0,
        lat=final_lat,
        lon=final_lon
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