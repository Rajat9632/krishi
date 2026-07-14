# Krishi Mitra

Krishi Mitra is a Flask-based agriculture assistant that helps users diagnose crop diseases, check live market prices, and ask farming questions through an AI assistant.

## Features

- Crop disease prediction powered by a hosted Hugging Face model endpoint
- Live market-price lookup using the public Agmarknet API
- Natural-language market queries like `orange in Karnataka`
- AI assistant chat page backed by a serverless Gemini endpoint
- Mobile-friendly UI with graceful fallback messages when external services are unavailable

## Tech Stack

- Python, Flask
- Requests, BeautifulSoup, Gradio client
- Google Generative AI
- HTML, CSS, and vanilla JavaScript

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the app with `python app.py` or your preferred Flask entrypoint.

## Deployment Notes

- The market-price feature now reads from the live Agmarknet API instead of scraping the old HTML table.
- External API failures return readable messages instead of breaking the page.
- Set `GEMINI_API_KEY` in your deployment environment for the AI assistant route.