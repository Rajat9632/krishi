import os
import json
from datetime import datetime
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow import keras
from keras.utils import custom_object_scope

import keras.backend as K
from keras.layers import Layer

# Load the API key from an environment variable
GEMINI_API_KEY = os.environ.get('AIzaSyDS-VZZyY8d-D9hrxM0gvbwktn_CoKlKMo')

class TrueDivide(Layer):
    def __init__(self, name=None, **kwargs):
        super(TrueDivide, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.math.truediv(x, y)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
from PIL import Image
import numpy as np
import google.generativeai as genai

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- 1. SET YOUR PATHS HERE ---
# ⚠️ IMPORTANT: Update this path to where you saved the dataset on YOUR computer.
# Use forward slashes '/'.
train_dir = 'naii/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train' 

# --- Generate Class Names ---
class_names = []
if os.path.exists(train_dir):
    class_names = sorted(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes.")
else:
    print(f"CRITICAL ERROR: The directory '{train_dir}' was not found. Please check the path in app.py.")

# --- 2. Load the AI Model for Image Prediction ---
try:
    with custom_object_scope({'TrueDivide': TrueDivide}):
        model = tf.keras.models.load_model('disease_model.keras')
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
    if model is None: return "Image prediction model failed to load. Check terminal for errors."
    if not class_names: return "Class names not loaded. Check dataset path in app.py."
    
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)
    pred_idx = np.argmax(predictions, axis=1)[0]
    pred_class = class_names[pred_idx]
    confidence = float(np.max(predictions))
    return f"Model Prediction: '{pred_class}' with {confidence:.2%} confidence."

# (The other functions for market prices remain the same)
state_codes = {"Karnataka": "KK", "Maharashtra": "MH"}
commodity_codes = {"Onion": "103", "Pigeon Pea (Tur)": "301", "Orange": "34"}

def fetch_market_prices(state_name="Karnataka", commodity_name="Orange"):
    """Fetch market prices with AI fallback when scraping fails or times out."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    date_str = datetime.now().strftime("%d-%b-%Y")
    state_code = state_codes.get(state_name, "0")
    commodity_code = commodity_codes.get(commodity_name, "0")
    from_date = "01-Jan-2024"
    to_date = date_str
    url = f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodity_code}&Tx_State={state_code}&Tx_District=0&Tx_Market=0&DateFrom={from_date}&DateTo={to_date}&Fr_Date={from_date}&To_Date={to_date}&Tx_Trend=0&Tx_CommodityHead={commodity_name}&Tx_StateHead={state_name}"
    results = []
    
    # Try scraping with timeout
    try:
        response = requests.get(url, headers=headers, timeout=5)  # 5 second timeout
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        data_table = soup.find('table', id='cphBody_GridPriceData')
        
        if data_table:
            rows = data_table.find_all('tr')[1:]
            if rows and "No Data Found" not in rows[0].text:
                for row in rows:
                    columns = row.find_all('td')
                    if len(columns) >= 8:
                        results.append({
                            'market': columns[3].text.strip(), 
                            'commodity': columns[4].text.strip(), 
                            'min_price': columns[6].text.strip(), 
                            'max_price': columns[7].text.strip()
                        })
                if results:
                    return results
    
    except requests.exceptions.Timeout:
        print("Scraping timed out, switching to AI fallback")
    except requests.exceptions.RequestException as e:
        print(f"Scraping failed: {e}")
    except Exception as e:
        print(f"Unexpected error during scraping: {e}")
    
    # Fallback to AI when scraping fails or times out
    print("Using AI fallback for market prices")
    return get_ai_market_prices(state_name, commodity_name)

def get_ai_market_prices(state_name, commodity_name):
    """Use Gemini AI to generate market prices when scraping fails."""
    try:
        genai.configure(api_key="AIzaSyDS-VZZyY8d-D9hrxM0gvbwktn_CoKlKMo")
        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        You are an agricultural market expert. Provide current market prices for {commodity_name} in {state_name}, India.
        Return exactly 3-4 sample market entries in JSON format like:
        [
            {{"market": "Market Name", "commodity": "{commodity_name}", "min_price": "₹XXX", "max_price": "₹XXX"}},
            {{"market": "Another Market", "commodity": "{commodity_name}", "min_price": "₹XXX", "max_price": "₹XXX"}}
        ]
        Use realistic prices based on current market trends. Return only the JSON array.
        """
        
        response = ai_model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        prices = json.loads(json_str)
        
        # Ensure we have proper data structure
        if isinstance(prices, list) and len(prices) > 0:
            return prices
        else:
            return [{'market': f'{state_name} Markets', 'commodity': commodity_name, 'min_price': '₹50', 'max_price': '₹80'}]
            
    except Exception as e:
        print(f"AI fallback failed: {e}")
        return [{'market': 'Data Unavailable', 'commodity': commodity_name, 'min_price': 'N/A', 'max_price': 'N/A'}]

def parse_query_with_ai(query):
    try:
        genai.configure(api_key="AIzaSyDS-VZZyY8d-D9hrxM0gvbwktn_CoKlKMo") # ⚠️ Remember to add your key!
    except Exception as e:
        return {'state': 'API Error', 'commodity': 'API Error'}
    prompt = f"""You are an NLU engine. Extract 'state' and 'commodity' from the user's query. Return a clean JSON object. If an entity is not found, use "Unknown". User Query: "{query}" JSON Result: """
    ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = ai_model.generate_content(prompt)
        json_response_str = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response_str)
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return {'state': 'AI Error', 'commodity': 'AI Error'}

# --- 4. All Routes (Webpages) ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file and file.filename != '':
        # Use the /tmp directory for file uploads
        filepath = os.path.join('/tmp', file.filename)
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

@app.route('/ai_query', methods=['GET', 'POST'])
def ai_query():
    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt', '')
        if not user_prompt:
            return render_template('ai_query.html', ai_response="Error: Please enter a prompt.")
        
        try:
            genai.configure(api_key="AIzaSyDS-VZZyY8d-D9hrxM0gvbwktn_CoKlKMo")
            ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            # Create a more comprehensive prompt for general AI assistance
            full_prompt = f"""You are a helpful agricultural AI assistant. Please provide a helpful and informative response to the following query. Keep your response concise but informative. Query: {user_prompt}"""
            
            response = ai_model.generate_content(full_prompt)
            ai_response = response.text.strip()
            
            return render_template('ai_query.html', ai_response=ai_response, user_prompt=user_prompt)
        except Exception as e:
            return render_template('ai_query.html', ai_response=f"Error generating AI response: {str(e)}", user_prompt=user_prompt)
    
    return render_template('ai_query.html')

# Vercel deployment
if __name__ == "__main__":
    app.run(debug=True)
else:
    application = app
