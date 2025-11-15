# Choice of LLM provider: 
# "gemini" for Google Gemini API
# "lmstudio" for LM Studio - locally hosted models
LLM_PROVIDER="gemini" # or "lmstudio"

# If you choose Gemini, input model name of choice. The full list can be found in the `model.txt`.
GEMINI_MODEL_NAME="models/gemini-2.5-flash" 
# Your API key for Gemini should already be in your .env file as:
# GEMINI_API_KEY="your_gemini_api_key_here"


# If you choose LM Studio, input the model name you have loaded in LM Studio.
# In this case, the default is set to "medgemma-4b-it". This is also the model's API identifier.
LMSTUDIO_MODEL="medgemma-4b-it"
# This is the default URL where LM Studio is hosted. You can double check in LM Studio Developer settings.
LMSTUDIO_URL="http://127.0.0.1:1234"
# Dictate how 'creative' the model's responses are. Higher values (e.g., 0.9) yield more diverse outputs, while lower values (e.g., 0.2) produce more focused and deterministic results.
LMSTUDIO_TEMPERATURE="0.7" 
# Set to 32K context window. Lower it if you want more performance or are limited by RAM.
LMSTUDIO_MAX_TOKENS=32768 
# Controls the cumulative probability for token selection. A value of 0.95 means the model considers tokens that make up 95% of the probability mass, balancing diversity and coherence.
LMSTUDIO_TOP_P=0.95