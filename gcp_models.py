from google import genai

PROJECT_ID = "search-ahmed"   # change if needed
LOCATION = "global"

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

print("\nAvailable Generative AI Models:\n")

for model in client.models.list():
    name = model.name

    # filter only LLM models
    if "gemini" in name.lower():
        print(name)