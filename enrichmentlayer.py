import os
import asyncio
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Fixes required for this to work at all:
# 1. User-Agent header — data.gov.in silently drops requests with Python's default UA (causes ReadTimeout hangs)
# 2. Correct resource_id + correct api_key (from your data.gov.in account's API page, not a placeholder)
CURL_UA = {"User-Agent": "curl/8.4.0"}
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


async def fetch_weather_metrics(lat: Optional[str], lon: Optional[str]) -> Dict[str, Any]:
    """
    Queries OpenWeatherMap to pull current humidity and temperature configurations.
    """
    # Fallback structure if coordinates are missing
    fallback = {"humidity": 60, "temperature": 27.0, "status": "coordinates_missing"}
    if not lat or not lon:
        return fallback

    api_key = os.environ.get("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=4.0)
            if response.status_code == 200:
                data = response.json()
                return {
                    "humidity": data.get("main", {}).get("humidity", 60),
                    "temperature": data.get("main", {}).get("temp", 27.0),
                    "status": "success"
                }
            logger.warning(f"Weather API returned status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching weather details: {str(e)}")
        
    return {"humidity": 60, "temperature": 27.0, "status": "api_error"}



async def reverse_geocode_state(lat: str, lon: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Reverse geocodes lat/lon to a state name using OpenStreetMap Nominatim (free, no API key).
    Returns the state name as reported by OSM, or None on failure.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    # Nominatim requires a descriptive User-Agent per their usage policy — reusing curl UA here would violate
    # their terms, so we identify our app honestly for this specific call.
    headers = {"User-Agent": "MandiPriceApp/1.0 (contact: gangasandeep222.com)"}

    try:
        r = await client.get(url, params=params, headers=headers, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            state = data.get("address", {}).get("state")
            logger.info(f"[geocode] lat={lat} lon={lon} -> state={state}")
            return state
        else:
            logger.warning(f"[geocode] non-200 response: {r.status_code}")
    except Exception as e:
        logger.error(f"[geocode] error: {type(e).__name__}: {repr(e)}")

    return None


async def _query_mandi(client: httpx.AsyncClient, api_key: str, commodity: str, state: Optional[str], limit: int) -> Optional[list]:
    """
    Single mandi query attempt with retries. Returns the records list, or None if the call ultimately failed/timed out.
    Returns an empty list [] if the call succeeded but had zero matching records (different from None = failure).
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[commodity]": commodity,
        "limit": str(limit),
    }
    if state:
        params["filters[state]"] = state

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            r = await client.get(BASE_URL, params=params, timeout=30.0)
            logger.info(f"[mandi] attempt={attempt} state={state} status={r.status_code}")

            if r.status_code == 200:
                res_json = r.json()
                records = res_json.get("records", [])
                logger.info(f"[mandi] state={state} count={res_json.get('count')} total={res_json.get('total')}")
                return records
            else:
                logger.error(f"[mandi] non-200 response: {r.status_code} body={r.text[:200]}")
                return None
        except httpx.TimeoutException:
            logger.error(f"[mandi] attempt={attempt} timed out")
            if attempt == max_retries:
                return None
            await asyncio.sleep(2 * attempt)
        except Exception as e:
            logger.error(f"[mandi] error: {type(e).__name__}: {repr(e)}")
            return None

    return None


async def fetch_mandi_prices(crop_name: str, lat: Optional[str] = None, lon: Optional[str] = None) -> Dict[str, Any]:
    """
    Queries the official Data.gov.in Agmarknet API endpoint for the given crop's mandi prices,
    filtered to the caller's state when lat/lon are provided (via reverse geocoding).
    Falls back to an unfiltered (nationwide) search if no records are found for that state.
    """
    fallback = {"modal_price_per_quintal": "Unavailable", "market_name": "Unknown", "status": "no_data"}
    api_key = os.environ.get("GOV_INDIA_API_KEY")

    if not api_key:
        logger.error("GOV_INDIA_API_KEY is not set in environment")
        return fallback

    if crop_name.lower() in ["unknown", "young plant", "healthy-plant"]:
        return fallback

    commodity = crop_name.capitalize()

    async with httpx.AsyncClient(headers=CURL_UA) as client:
        # Step 1: figure out the caller's state, if coordinates are available
        state = None
        if lat and lon:
            state = await reverse_geocode_state(lat, lon, client)

        # Step 2: try state-filtered query first (if we have a state)
        if state:
            records = await _query_mandi(client, api_key, commodity, state, limit=10)
            if records:
                best = records[0]
                return {
                    "modal_price_per_quintal": best.get("modal_price", "Unavailable"),
                    "market_name": best.get("market", "Local Mandi"),
                    "district": best.get("district", "Unknown"),
                    "state": best.get("state", state),
                    "status": "success"
                }
            elif records == []:
                logger.warning(f"[mandi] no records for commodity='{commodity}' in state='{state}', broadening search")
            else:
                # records is None -> the call itself failed after retries
                return fallback

        # Step 3: fallback to nationwide search (no state filter) if state-filtered search found nothing
        records = await _query_mandi(client, api_key, commodity, state=None, limit=5)
        if records:
            best = records[0]
            return {
                "modal_price_per_quintal": best.get("modal_price", "Unavailable"),
                "market_name": best.get("market", "Local Mandi"),
                "district": best.get("district", "Unknown"),
                "state": best.get("state", "Unknown"),
                "status": "success_nationwide_fallback"  # flag that this wasn't localized to the user's state
            }

    return fallback
# async def fetch_mandi_prices(crop_name: str, lat: Optional[str] = None, lon: Optional[str] = None) -> Dict[str, Any]:
    """
    Queries the official Data.gov.in Agmarknet API endpoint.
    Filters market arrivals based on the parsed crop identity.
    """
    fallback = {"modal_price_per_quintal": "Unavailable", "market_name": "Unknown", "status": "no_data"}
    api_key = os.environ.get("GOV_INDIA_API_KEY")

    if not api_key:
        logger.error("GOV_INDIA_API_KEY is not set in environment")
        return fallback

    if crop_name.lower() in ["unknown", "young plant", "healthy-plant"]:
        return fallback

    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    base_url = f"https://api.data.gov.in/resource/{resource_id}"

    async with httpx.AsyncClient() as client:
        url = f"{base_url}?api-key={api_key}&format=json&filters[commodity]={crop_name.capitalize()}&limit=1"

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.get(url, timeout=5)
                logger.info(f"[mandi] attempt={attempt} status={response.status_code}")

                if response.status_code == 200:
                    res_json = response.json()
                    records = res_json.get("records", [])
                    if records:
                        latest_record = records[0]
                        return {
                            "modal_price_per_quintal": latest_record.get("modal_price", "Unavailable"),
                            "market_name": latest_record.get("market", "Local Mandi"),
                            "state": latest_record.get("state", "Unknown"),
                            "status": "success"
                        }
                    else:
                        logger.warning(f"[mandi] no records for commodity='{crop_name.capitalize()}'")
                        break
                else:
                    logger.error(f"[mandi] non-200 response: {response.status_code}")
                    break
            except httpx.TimeoutException as e:
                logger.error(f"[mandi] attempt={attempt} timed out: {type(e).__name__}")
                if attempt == max_retries:
                    return fallback
                await asyncio.sleep(2 * attempt)
            except Exception as e:
                logger.error(f"[mandi] error: {type(e).__name__}: {repr(e)}")
                return fallback

    return fallback
# async def fetch_mandi_prices(crop_name: str, lat: Optional[str] = None, lon: Optional[str] = None) -> Dict[str, Any]:
#     """
#     Queries the official Data.gov.in Agmarknet API endpoint.
#     Filters market arrivals based on the parsed crop identity.
#     """
#     fallback = {"modal_price_per_quintal": "Unavailable", "market_name": "Unknown", "status": "no_data"}
#     api_key = os.environ.get("GOV_INDIA_API_KEY")
 
#     if not api_key:
#         logger.error("GOV_INDIA_API_KEY is not set in environment")
#         return fallback
 
#     if crop_name.lower() in ["unknown", "young plant", "healthy-plant"]:
#         return fallback
 
#     # FIXED: correct resource id (previous one had a typo: ...a86d9fb643ff vs correct ...a864a43d0070)
#     resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
 
#     base_url = f"https://api.data.gov.in/resource/{resource_id}"
 
#     async with httpx.AsyncClient() as client:
#         # --- Attempt 1: filtered by exact commodity name ---
#         url = f"{base_url}?api-key={api_key}&format=json&filters[commodity]={crop_name.capitalize()}&limit=1"
#         try:
#             response = await client.get(url, timeout=5.0)
#             logger.info(f"[mandi] status={response.status_code} url_no_key={url.replace(api_key, 'REDACTED')}")
 
#             if response.status_code == 200:
#                 res_json = response.json()
#                 logger.info(f"[mandi] response meta: message={res_json.get('message')}, count={res_json.get('count')}, total={res_json.get('total')}")
 
#                 records = res_json.get("records", [])
#                 if records:
#                     latest_record = records[0]
#                     return {
#                         "modal_price_per_quintal": latest_record.get("modal_price", "Unavailable"),
#                         "market_name": latest_record.get("market", "Local Mandi"),
#                         "state": latest_record.get("state", "Unknown"),
#                         "status": "success"
#                     }
#                 else:
#                     logger.warning(f"[mandi] no records for commodity filter='{crop_name.capitalize()}'. "
#                                    f"Message from API: {res_json.get('message')}")
#             else:
#                 logger.error(f"[mandi] non-200 response: {response.status_code} body={response.text[:300]}")
#         except Exception as e:
#             logger.error(f"[mandi] error querying with commodity filter: {str(e)}")
#             return fallback
 
#         # --- Attempt 2: no filter, just to sanity-check the endpoint/resource id/key are valid ---
#         # If this also returns zero records, the problem is resource_id/api_key, not your crop name.
#         sanity_url = f"{base_url}?api-key={api_key}&format=json&limit=5"
#         try:
#             response = await client.get(sanity_url, timeout=5.0)
#             if response.status_code == 200:
#                 res_json = response.json()
#                 records = res_json.get("records", [])
#                 if records:
#                     sample_commodities = [r.get("commodity") for r in records]
#                     logger.warning(f"[mandi] endpoint IS working. Sample commodity names in data: {sample_commodities}. "
#                                    f"Your crop_name='{crop_name}' likely doesn't match these strings exactly.")
#                 else:
#                     logger.error(f"[mandi] endpoint returned zero records even unfiltered. "
#                                  f"Check resource_id and api_key. Message: {res_json.get('message')}")
#         except Exception as e:
#             logger.error(f"[mandi] error on sanity check call: {str(e)}")
 
#     return fallback
# async def fetch_mandi_prices(crop_name: str, lat: Optional[str] = None, lon: Optional[str] = None) -> Dict[str, Any]:
#     """
#     Queries the official Data.gov.in Agmarknet API endpoint.
#     Filters market arrivals based on the parsed crop identity.
#     """
#     fallback = {"modal_price_per_quintal": "Unavailable", "market_name": "Unknown", "status": "no_data"}
#     api_key = os.environ.get("GOV_INDIA_API_KEY")
 
#     if not api_key:
#         logger.error("GOV_INDIA_API_KEY is not set in environment")
#         return fallback
 
#     if crop_name.lower() in ["unknown", "young plant", "healthy-plant"]:
#         return fallback
 
#     # FIXED: correct resource id (previous one had a typo: ...a86d9fb643ff vs correct ...a864a43d0070)
#     resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
 
#     base_url = f"https://api.data.gov.in/resource/{resource_id}"
 
#     async with httpx.AsyncClient() as client:
#         # --- Attempt 1: filtered by exact commodity name ---
#         url = f"{base_url}?api-key={api_key}&format=json&filters[commodity]={crop_name.capitalize()}&limit=1"
#         try:
#             response = await client.get(url, timeout=15.0)
#             logger.info(f"[mandi] status={response.status_code} url_no_key={url.replace(api_key, 'REDACTED')}")
 
#             if response.status_code == 200:
#                 res_json = response.json()
#                 logger.info(f"[mandi] response meta: message={res_json.get('message')}, count={res_json.get('count')}, total={res_json.get('total')}")
 
#                 records = res_json.get("records", [])
#                 if records:
#                     latest_record = records[0]
#                     return {
#                         "modal_price_per_quintal": latest_record.get("modal_price", "Unavailable"),
#                         "market_name": latest_record.get("market", "Local Mandi"),
#                         "state": latest_record.get("state", "Unknown"),
#                         "status": "success"
#                     }
#                 else:
#                     logger.warning(f"[mandi] no records for commodity filter='{crop_name.capitalize()}'. "
#                                    f"Message from API: {res_json.get('message')}")
#             else:
#                 logger.error(f"[mandi] non-200 response: {response.status_code} body={response.text[:300]}")
#         except Exception as e:
#             logger.error(f"[mandi] error querying with commodity filter: {type(e).__name__}: {repr(e)}")
#             return fallback
 
#         # --- Attempt 2: no filter, just to sanity-check the endpoint/resource id/key are valid ---
#         # If this also returns zero records, the problem is resource_id/api_key, not your crop name.
#         sanity_url = f"{base_url}?api-key={api_key}&format=json&limit=5"
#         try:
#             response = await client.get(sanity_url, timeout=15.0)
#             if response.status_code == 200:
#                 res_json = response.json()
#                 records = res_json.get("records", [])
#                 if records:
#                     sample_commodities = [r.get("commodity") for r in records]
#                     logger.warning(f"[mandi] endpoint IS working. Sample commodity names in data: {sample_commodities}. "
#                                    f"Your crop_name='{crop_name}' likely doesn't match these strings exactly.")
#                 else:
#                     logger.error(f"[mandi] endpoint returned zero records even unfiltered. "
#                                  f"Check resource_id and api_key. Message: {res_json.get('message')}")
#         except Exception as e:
#             logger.error(f"[mandi] error on sanity check call: {type(e).__name__}: {repr(e)}")
 
#     return fallback