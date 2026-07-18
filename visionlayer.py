import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os

client = genai.Client()

class PlantDiagnostic(BaseModel):
    is_valid_crop_image: bool = Field(description="True if the image contains plant leaves/crops.")
    crop_name: str = Field(description="Common name of the crop (e.g., Tomato).")
    disease_detected: str = Field(description="Name of the identified disease.")
    confidence_score: float = Field(description="Confidence value between 0.0 and 1.0.")
    visual_symptoms: List[str] = Field(description="List of observed symptoms.")
    search_query_key: str = Field(description="Hyphenated lookup key (e.g., 'early-blight').")

async def analyze_crop_image(media_url: str) -> PlantDiagnostic:
    """
    Downloads the protected image from Twilio using HTTP Basic Authentication
    and processes it via Gemini.
    """
    # 1. Pull Twilio credentials that were loaded by load_dotenv() in main.py
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    
# Fetch the image and explicitly tell httpx to follow Twilio's redirect links
    async with httpx.AsyncClient() as async_client:
        response = await async_client.get(
            media_url, 
            follow_redirects=True  # <-- This fixes the 307 Redirect issue!
        )
        if response.status_code != 200:
            raise Exception(f"Twilio proxy download failed with status code: {response.status_code}")
        image_bytes = response.content

    # 3. Package and send to Gemini
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    prompt = "Analyze this agricultural leaf specimen photograph for any diseases."

    api_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlantDiagnostic,
            temperature=0.1
        )
    )

    return api_response.parsed