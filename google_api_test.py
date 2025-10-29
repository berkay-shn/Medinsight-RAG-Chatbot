import google.generativeai as genai
from dotenv import load_dotenv
import os

print("--- Google AI Model Authorization Check Started ---")

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("\nERROR: GOOGLE_API_KEY not found in .env file. Please add your API key.")
else:
    try:
        # Configure Google GenAI API key
        genai.configure(api_key=api_key)
        print("\nAPI key configured successfully.")
        print("Listing available models...")

        model_count = 0
        for m in genai.list_models():
            # List only models that support the 'generateContent' method
            if 'generateContent' in m.supported_generation_methods:
                print(f"- Model Name: {m.name}")
                model_count += 1

        if model_count == 0:
            print("\nWARNING: No models supporting the 'generateContent' method were found for this API key.")
            print("Please check your Google Cloud project settings and API permissions.")
        else:
            print(f"\nSUCCESS: Total {model_count} available models found.")

    except Exception as e:
        print("\n!!! ERROR: An issue occurred while connecting to the Google GenAI API. !!!")
        print("Possible reasons: Invalid API key or incomplete Google Cloud project settings.")
        print("Original Error Message:")
        print(e)

print("\n--- Check Completed ---")