import os
from dotenv import load_dotenv
from google import genai

# 1. Load the environment variables from the .env file in the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_root, ".env")

print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)

api_key = os.environ.get("GEMINI_API_KEY")
print(f"GEMINI_API_KEY present in environment: {api_key is not None}")

if not api_key:
    print("Error: GEMINI_API_KEY is missing! Make sure your .env file exists and contains GEMINI_API_KEY=your_key")
    exit(1)

# 2. Initialize the client (it will now automatically find the GEMINI_API_KEY from .env)
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Verify connection"
)

print("Success! Gemini response:", response.text.strip())


