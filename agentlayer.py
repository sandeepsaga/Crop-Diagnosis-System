import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from google import genai
from dotenv import load_dotenv
load_dotenv()

# Define the state shape flowing through the graph channels
class AgentState(TypedDict):
    vision_data: Dict[str, Any]
    weather_data: Dict[str, Any]
    mandi_data: Dict[str, Any]
    economic_directive: str
    final_treatment_plan: str

# Initialize the client specifically with the dedicated Agent API Key
agent_key = os.environ.get("GEMINI_AGENT_KEY")
agent_client = genai.Client(api_key=agent_key)

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes market data constraints to establish a clear budget framework for the advice.
    """
    crop = state["vision_data"].get("crop_name")
    mandi = state["mandi_data"]
    price = mandi.get("modal_price_per_quintal")
    
    prompt = f"""
    You are the Economic Coordinator Agent for an agricultural ecosystem.
    The current crop being analyzed is: {crop}.
    The local market price context per quintal is: {price} INR.

    Based on the market price availability, generate a short 2-sentence strategy directive for the advisory team.
    - If the price is 'Unavailable', instruct them to focus purely on standard, resource-efficient care tips.
    - If the price is low, instruct them to prioritize low-cost, homemade organic solutions.
    - If the price is strong, instruct them to recommend aggressive, high-yield protection tactics.
    """
    
    response = agent_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    return {"economic_directive": response.text.strip()}


def advisory_node(state: AgentState) -> Dict[str, Any]:
    """
    Blends the disease analysis, weather data, and economic limits to craft a localized guide.
    """
    vision = state["vision_data"]
    weather = state["weather_data"]
    directive = state["economic_directive"]
    
    prompt = f"""
    You are an Expert Agronomist Advisor speaking directly to a farmer.
    
    --- FIELD DIAGNOSIS ---
    Crop Type: {vision.get('crop_name')}
    Health Status: {vision.get('disease_detected')}
    Symptoms: {vision.get('visual_symptoms')}
    
    --- LOCAL ENVIRONMENT ---
    Temperature: {weather.get('temperature')}°C
    Humidity Level: {weather.get('humidity')}%
    
    --- STRATEGIC BUDGET DIRECTIVE ---
    {directive}
    
    Write a highly targeted advice report for this farmer. 
    Keep your vocabulary simple, direct, and actionable. 
    If the plant is healthy, tell them how to maintain it and optimize its growth. 
    If a disease is present, provide clear step-by-step mitigation options within the budget directive constraints.
    """
    
    response = agent_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    return {"final_treatment_plan": response.text.strip()}

def run_agentic_reasoning(vision_res: Any, weather_res: Dict[str, Any], mandi_res: Dict[str, Any]) -> str:
    """
    Assembles the nodes into a compiled execution graph and processes the input data payload.
    """
    builder = StateGraph(AgentState)
    
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("adviser", advisory_node)
    
    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "adviser")
    builder.add_edge("adviser", END)
    
    graph = builder.compile()
    
    initial_inputs = {
        "vision_data": vision_res if isinstance(vision_res, dict) else vision_res.model_dump(),
        "weather_data": weather_res,
        "mandi_data": mandi_res,
        "economic_directive": "",
        "final_treatment_plan": ""
    }
    
    final_state = graph.invoke(initial_inputs)
    return final_state["final_treatment_plan"]