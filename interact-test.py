# Testing interaction via API

import google.generativeai as genai
import os

# --- PASTE YOUR API KEY HERE ---
# You can also set it as an environment variable named "GEMINI_API_KEY"
API_KEY = "YOUR_API_KEY"

# Configure the library with your API key
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring the API. Please check your API key. Details: {e}")
    exit()

# Select the model you want to use
model = genai.GenerativeModel('gemini-2.5-pro') # A fast and efficient model

# --- Main Program Loop ---
def main():
    """
    This function runs a loop to continuously get user input
    and query the Gemini API.
    """
    print("Connected to Gemini. You can start asking questions.")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        # Get input from the user
        prompt = input("\nYou: ")

        if prompt.lower() in ['quit', 'exit']:
            print("Exiting the program. Goodbye!")
            break

        if not prompt:
            print("Please enter a question.")
            continue

        # Send the prompt to the model and get the response
        try:
            response = model.generate_content(prompt)
            print(f"\nGemini: {response.text}")
        except Exception as e:
            print(f"An error occurred while getting the response: {e}")

# Run the main function
if __name__ == '__main__':
    if API_KEY == "YOUR_API_KEY":
        print("ERROR: Please replace 'YOUR_API_KEY' with your actual Gemini API key in the script.")
    else:
        main()

