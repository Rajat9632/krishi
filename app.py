import os
import json
from datetime import datetime
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from gradio_client import Client

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

# ⚠️ IMPORTANT: Paste your public Hugging Face Space URL here
HUGGING_FACE_URL = "RajatChoudhary/krishi-mitra-model"

def get_real_prediction(image_path):
    try:
        client = Client(HUGGING_FACE_URL)
        
        # Open the file so Gradio can upload it
        with open(image_path, "rb") as f:
            result = client.predict(
                f,  # Pass file object, not just path
                api_name="/predict"
            )

        if "confidences" not in result:
            return "Error: Unexpected response from the model."

        confidences = result["confidences"]
        if confidences and isinstance(confidences[0], list):
            confidences = [{"label": c[0], "confidence": c[1]} for c in confidences]

        top_prediction = max(confidences, key=lambda x: x["confidence"])
        return f"Model Prediction: '{top_prediction['label']}' with {top_prediction['confidence']:.2%} confidence."

    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Error: Could not get a prediction from the AI model. The model may be waking up. Please try again in a minute."



# (All other functions for market prices, etc., remain the same)
state_codes = {"Karnataka": "KK", "Maharashtra": "MH"}
commodity_codes = {"Onion": "103", "Pigeon Pea (Tur)": "301", "Orange": "34"}

def fetch_market_prices(state_name="Karnataka", commodity_name="Orange"):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    date_str = datetime.now().strftime("%d-%b-%Y")
    state_code = state_codes.get(state_name, "0")
    commodity_code = commodity_codes.get(commodity_name, "0")
    from_date = "01-Jan-2024"
    to_date = date_str
    url = f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodity_code}&Tx_State={state_code}&Tx_District=0&Tx_Market=0&DateFrom={from_date}&DateTo={to_date}&Fr_Date={from_date}&To_Date={to_date}&Tx_Trend=0&Tx_CommodityHead={commodity_name}&Tx_StateHead={state_name}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        data_table = soup.find('table', id='cphBody_GridPriceData')
        if not data_table: return [{'market': 'Error', 'commodity': 'Data table not found.'}]
        rows = data_table.find_all('tr')[1:]
        if not rows or "No Data Found" in rows[0].text: return [{'market': 'No Data', 'commodity': 'No data found for this query.'}]
        results = []
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 8:
                results.append({'market': columns[3].text.strip(), 'commodity': columns[4].text.strip(), 'min_price': columns[6].text.strip(), 'max_price': columns[7].text.strip()})
        return results
    except Exception as e:
        return [{'market': 'Scraper Failed', 'commodity': 'Could not fetch live data.'}]

def parse_query_with_keywords(query):
    query = query.lower()
    state = "Unknown"; commodity = "Unknown"
    if "karnataka" in query: state = "Karnataka"
    if "maharashtra" in query: state = "Maharashtra"
    if "tur" in query or "pigeon pea" in query: commodity = "Pigeon Pea (Tur)"
    if "onion" in query: commodity = "Onion"
    if "orange" in query: commodity = "Orange"
    return {'state': state, 'commodity': commodity}

@app.route('/')
def homepage(): return render_template('index.html')

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
    if not user_query: return render_template('result.html', prediction_text="Error: You did not enter a query.")
    entities = parse_query_with_keywords(user_query)
    state = entities.get('state', 'Unknown')
    commodity = entities.get('commodity', 'Unknown')
    price_data = fetch_market_prices(state_name=state, commodity_name=commodity)
    return render_template('prices.html', price_data=price_data)