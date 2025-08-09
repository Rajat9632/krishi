import os
import json
from datetime import datetime
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai

# --- App Initialization ---
app = Flask(__name__)
# Use Vercel's temporary directory for uploads
app.config['UPLOAD_FOLDER'] = '/tmp'

# --- 1. API Key and Dataset Path Configuration ---

# Load the API key from an environment variable
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not found.")

# --- Load Class Names from JSON File ---
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    print(f"Successfully loaded {len(class_names)} class names.")
except FileNotFoundError:
    print("CRITICAL ERROR: 'class_names.json' not found. Please add it to your repository.")
    class_names = []

# --- Generate Class Names ---
class_names = []
if os.path.exists(train_dir):
    class_names = sorted(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes.")
else:
    print(f"CRITICAL ERROR: The directory '{train_dir}' was not found. Please check the path in app.py.")

# --- 2. Load the Custom AI Model ---
try:
    # Load the model without the training configuration
    model = tf.keras.models.load_model('disease_model.h5', compile=False)
    print("Custom disease detection model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    model = None

# --- 3. All Function Definitions ---

def process_image(image_path):
    """Prepares an image for our custom model by only resizing it."""
    img = Image.open(image_path).resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return img_array_expanded

def get_real_prediction(image_path):
    """Uses the loaded custom model to make a prediction."""
    if model is None:
        return "Image prediction model failed to load. Check server logs."
    if not class_names:
        return "Class names not loaded. Check dataset path in app.py."
    
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)
    pred_idx = np.argmax(predictions, axis=1)[0]
    pred_class = class_names[pred_idx]
    confidence = float(np.max(predictions))
    return f"Model Prediction: '{pred_class}' with {confidence:.2%} confidence."

def get_ai_market_prices(state_name, commodity_name):
    """Fallback: Use Gemini AI to generate market prices."""
    try:
        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Provide a few realistic, current market prices for {commodity_name} in {state_name}, India. Return as a JSON array of objects with keys 'market', 'commodity', 'min_price', 'max_price'. Return only the JSON array."
        response = ai_model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_str)
    except Exception as e:
        print(f"AI fallback for prices failed: {e}")
        return [{'market': 'Data Unavailable', 'commodity': commodity_name, 'min_price': 'N/A', 'max_price': 'N/A'}]

def fetch_market_prices(state_name="Karnataka", commodity_name="Orange"):
    """Fetch market prices, with an AI fallback if scraping fails."""
    # (Your scraper code here... for now, we will just use the AI fallback)
    print("Using AI fallback for market prices")
    return get_ai_market_prices(state_name, commodity_name)

def parse_query_with_ai(query):
    """Use Gemini to extract entities from a user's query."""
    try:
        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""You are an NLU engine. Extract 'state' and 'commodity' from this query. Return a clean JSON object. If an entity is not found, use "Unknown". Query: "{query}" JSON Result: """
        response = ai_model.generate_content(prompt)
        json_response_str = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response_str)
    except Exception as e:
        print(f"An error occurred during NLU API call: {e}")
        return {'state': 'AI Error', 'commodity': 'AI Error'}

# --- 4. All Routes (Webpages) ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        prediction = get_real_prediction(filepath)
        return render_template('result.html', prediction_text=prediction)
    return render_template('result.html', prediction_text="Error: No file selected.")

@app.route('/market-prices')
def market_prices():
    price_data = fetch_market_prices()
    return render_template('prices.html', price_data=price_data)

@app.route('/query-price', methods=['POST'])
def query_price():
    user_query = request.form.get('user_query', '')
    if not user_query:
        return render_template('result.html', prediction_text="Error: You did not enter a query.")
    
    entities = parse_query_with_ai(user_query)
    state = entities.get('state', 'Unknown')
    commodity = entities.get('commodity', 'Unknown')
    
    price_data = fetch_market_prices(state_name=state, commodity_name=commodity)
    return render_template('prices.html', price_data=price_data)

@app.route('/ai-query')
def ai_query_page():
    return render_template('ai_query.html')

@app.route('/ai-response', methods=['POST'])
def ai_response():
    user_prompt = request.form.get('user_prompt', '')
    if not user_prompt:
        return render_template('ai_query.html', ai_response="Error: Please enter a prompt.")
    try:
        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = ai_model.generate_content(user_prompt)
        return render_template('ai_query.html', ai_response=response.text, user_prompt=user_prompt)
    except Exception as e:
        return render_template('ai_query.html', ai_response=f"Error: {e}", user_prompt=user_prompt)