import google.generativeai as genai
import os

# --- PASTE YOUR API KEY HERE ---
# This is the same key you used before.
API_KEY = "AIzaSyDplDb3lir6NPq3GNwTAAKvlOmJv5as9lg"

# Configure the library with your API key
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring the API. Please check your API key. Details: {e}")
    exit()

print("Fetching available models for your API key...\n")

# List all models and check which ones support the 'generateContent' method
try:
    print("Models that support text generation (for your chat script):")
    print("-" * 55)
    for m in genai.list_models():
        # We check if the model supports the method used for chatting
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    print("-" * 55)

except Exception as e:
    print(f"An error occurred while fetching models: {e}")
