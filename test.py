import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get your Google API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY","AIzaSyDlmr8oCCeI_aLIVBdyyClL7euUkGCfT80")

if not API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it in your .env file or directly in your system's environment.")
    exit()

# Configure the generative AI client with your API key
genai.configure(api_key="AIzaSyDlmr8oCCeI_aLIVBdyyClL7euUkGCfT80")

print("Attempting to list models available via your API key...")
print("-" * 50)

try:
    # Iterate through all models accessible with your API key
    for m in genai.list_models():
        # Print basic model information
        print(f"Model Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Description: {m.description}")
        print(f"  Input Modalities: {m.input_token_limit}") # This seems incorrect based on documentation, should be input_modalities
        print(f"  Output Token Limit: {m.output_token_limit}")
        
        # Supported generation methods are important for knowing what the model can do
        print(f"  Supported Generation Methods: {m.supported_generation_methods}")
        
        # Check if the model supports 'generateContent', which is typically used for chat/text generation
        if 'generateContent' in m.supported_generation_methods:
            print("  --> This model supports text/chat generation.")
        print("-" * 50)

except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("This often indicates an issue with your API key (invalid, revoked, or not provisioned for the Generative Language API).")
    print("Please ensure your GOOGLE_API_KEY is correct and the Generative Language API is enabled in your Google Cloud Project.")