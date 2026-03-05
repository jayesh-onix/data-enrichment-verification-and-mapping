import asyncio
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PROJECT_ID = "search-ahmed"
LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"

class VerificationResponse(BaseModel):
    website_correct: bool = Field(description="Is the provided website the official primary website?")
    hq_state_correct: bool = Field(description="Is the HQ State correct?")
    description_correct: bool = Field(description="Does the description accurately describe the company?")
    region_correct: bool = Field(description="Is the region consistent with HQ State/country?")
    industry_reasonable: bool = Field(description="Is the industry classification reasonable?")
    annual_revenue_reasonable: bool = Field(description="Is the annual revenue plausible/reasonable?")

    corrected_website: str | None = Field(default=None, description="If incorrect, the official website to use")
    corrected_hq_state: str | None = Field(default=None, description="If incorrect, corrected full state/province")
    corrected_region: str | None = Field(default=None, description="If incorrect, corrected region")
    notes: str = Field(description="Brief evidence-based notes (include sources if possible)")

def build_verification_prompt() -> str:
    schema_json = json.dumps(VerificationResponse.model_json_schema(), indent=2)
    return f"""You are a data quality auditor with access to Google Search.
Verify the provided company fields against web evidence (official site, LinkedIn, reputable sources).

Company:
- Account Name: Google

Provided fields to verify:
- Website: google.com
- HQ State: CA
- Region: Americas
- Industry: Tech
- Annual Revenue (USD digits): 1000000
- Segment: Enterprise
- Description: Search engine

Rules:
1) Use Google Search grounding. Prefer official website and LinkedIn company page.
2) Mark each field as correct/incorrect/reasonable using booleans in the JSON schema.
3) If a field is incorrect, provide a corrected value (when possible).
4) If uncertain, be conservative: set the boolean to false and explain why in notes.
5) Notes must be short and evidence-based; include sources if you can.

Return ONLY valid JSON matching the schema below. Do NOT wrap in markdown code blocks.
{schema_json}
"""

async def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0,
    )
    prompt = build_verification_prompt()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        print("Success:", response.text)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
