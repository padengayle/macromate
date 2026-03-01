import base64
from openai import OpenAI
from dotenv import load_dotenv
from schemas import GuardrailResult, MealResponse, SafetyResult

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# In production, this should be injected as a dependency
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

def run_meal_analysis(image_path: str, model: str = "gpt-4o-mini") -> MealResponse:
    """
    Layer 2: Core Analysis.
    Extracts macros and ingredients. 
    Note: Defaults to 'gpt-4o-mini' as A/B testing showed it has comparable 
    accuracy to 'gpt-4o' but is ~30x cheaper and 3x faster.
    """
    base64_image = encode_image(image_path)
    
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": """
                You are a nutritional AI assistant. Analyze the provided meal image.
                1. Estimate macros (calories, carbs, fat, protein) based on visual portion sizes.
                2. List ingredients and assess their glycemic impact (green/yellow/orange/red).
                3. Provide a neutral, objective description.
                4. Do NOT provide medical diagnoses.
                """
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

def run_safety_check(meal_response: MealResponse, model: str = "gpt-4o-mini") -> SafetyResult:
    """
    Layer 3: Safety/Compliance.
    Scans the generated text to prevent medical hallucinations (e.g., insulin advice).
    Decouples 'analysis' from 'safety' for stricter control.
    """
    text_to_check = f"""
    Title: {meal_response.title}
    Description: {meal_response.description}
    Guidance: {meal_response.guidance}
    """
    
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": """
                You are a safety officer. Review the text for safety violations. 
                Flag True if it contains:
                - Emotional/judgmental language (e.g. "disgusting", "bad for you")
                - Risky substitutions (e.g. "stop taking insulin")
                - Medical diagnosis or treatment recommendations
                """
            },
            {"role": "user", "content": text_to_check}
        ],
        response_format=SafetyResult,
    )
    return response.choices[0].message.parsed