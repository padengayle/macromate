import base64
import asyncio
import json
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from schemas import GuardrailResult, MealResponse, SafetyResult

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def encode_image(image_path: str) -> str:
    """Encodes a local image to Base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_guardrails(image_path: str, model: str = "gpt-4o-mini") -> GuardrailResult:
    """
    Layer 1: Input Guardrail.
    Uses a cheap model to reject non-food/PII before expensive processing.
    """
    base64_image = encode_image(image_path)
    
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are a content moderator. Your task is to strictly flag any PII, humans, or captchas. Confirm if the image contains food."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        response_format=GuardrailResult,
    )
    return response.choices[0].message.parsed

def run_meal_analysis(image_path: str, model: str = "gpt-4o") -> MealResponse:
    """
    Layer 2: Core Extraction.
    Evaluates portion sizes and estimates macros from visual ingredients.
    """
    base64_image = encode_image(image_path)
    
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are an expert nutritionist. Analyze the meal, extract ingredients, and estimate macros."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        response_format=MealResponse,
    )
    return response.choices[0].message.parsed


# --- THE NEW MCP CLIENT INTEGRATION ---

async def async_run_safety_check(meal_response: MealResponse, user_id: str, model: str = "gpt-4o-mini") -> SafetyResult:
    """
    Layer 3: Safety/Compliance with Secure Enterprise Data.
    Connects to the local MCP Server, gets the user context, and evaluates safety.
    """
    # 1. Connect to our local MCP server script
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2. Call the secure MCP tool to get the user's health profile
            context_result = await session.call_tool("get_user_health_context", arguments={"user_id": user_id})
            patient_context = context_result.content[0].text
            
            # 3. Inject the secure context into the safety prompt
            text_to_check = f"""
            PATIENT HEALTH PROFILE (Strictly enforce allergies/conditions):
            {patient_context}
            
            MEAL TO EVALUATE:
            Title: {meal_response.title}
            Description: {meal_response.description}
            Guidance: {meal_response.guidance}
            """
            
            # 4. Run the standard safety check armed with private data
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        You are a safety officer. Review the meal against the patient's health profile.
                        Flag True if the meal violates their allergies or medical conditions.
                        Flag True if it contains medical diagnosis or treatment recommendations.
                        """
                    },
                    {"role": "user", "content": text_to_check}
                ],
                response_format=SafetyResult,
            )
            return response.choices[0].message.parsed

def run_safety_check(meal_response: MealResponse, user_id: str, model: str = "gpt-4o-mini") -> SafetyResult:
    """Synchronous wrapper for main pipeline execution."""
    return asyncio.run(async_run_safety_check(meal_response, user_id, model))
