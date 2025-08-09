import google.generativeai as genai

# Paste your same API key here
API_KEY = "AIzaSyDS-VZZyY8d-D9hrxM0gvbwktn_CoKlKMo"

print("Attempting to connect to Google AI...")

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = "What is the capital of Karnataka?"
    response = model.generate_content(prompt)

    print("\n--- SUCCESS ---")
    print(response.text)

except Exception as e:
    print("\n--- FAILED ---")
    print("An error occurred. This means the API key is invalid, the API is not enabled, or there is a network issue.")
    print(f"\nDetailed Error: {e}")