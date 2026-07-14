import mimetypes
import os
import shutil
from datetime import datetime, timedelta
from functools import lru_cache

import requests
from flask import Flask, render_template, request
from gradio_client import Client
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

HUGGING_FACE_URL = "RajatChoudhary/krishi-mitra-model"
AGMARKNET_API_BASE = "https://api.agmarknet.gov.in/v1"
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}


def _error_row(message):
    return {
        'status': 'error',
        'message': message,
    }


def _clean_text(value):
    return str(value).strip().lower()


def _pick_first(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, '', [], {}):
            return value
    return None


def _normalize_price_value(value):
    if value in (None, ''):
        return ''
    if isinstance(value, (int, float)):
        return f'{value:,.0f}'
    text = str(value).strip()
    try:
        return f'{float(text):,.0f}'
    except ValueError:
        return text


@lru_cache(maxsize=1)
def _load_agmarknet_filters():
    try:
        response = requests.get(
            f'{AGMARKNET_API_BASE}/daily-price-arrival/filters',
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get('status') and isinstance(payload.get('data'), dict):
            return payload['data']
    except Exception as exc:
        app.logger.warning('Failed to load Agmarknet filters: %s', exc)
    return {}


def _find_id(entries, name_keys, wanted_name, id_keys):
    if not wanted_name:
        return None

    normalized_wanted = _clean_text(wanted_name)
    if not normalized_wanted:
        return None

    exact_matches = []
    fallback_matches = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        candidate_name = None
        for key in name_keys:
            value = entry.get(key)
            if value:
                candidate_name = str(value)
                break

        if not candidate_name:
            continue

        candidate_normalized = _clean_text(candidate_name)
        candidate_id = None
        for key in id_keys:
            value = entry.get(key)
            if value is not None:
                candidate_id = value
                break

        if candidate_id is None:
            continue

        if candidate_normalized == normalized_wanted:
            exact_matches.append(candidate_id)
        elif normalized_wanted in candidate_normalized:
            fallback_matches.append(candidate_id)

    if exact_matches:
        return exact_matches[0]
    if fallback_matches:
        return fallback_matches[0]
    return None


def _normalize_report_row(row):
    if isinstance(row, dict):
        market = _pick_first(row, ['market_name', 'market', 'mkt_name', 'marketName']) or 'Unknown market'
        commodity = _pick_first(row, ['commodity_name', 'commodity', 'cmdt_name', 'commodityName']) or 'Unknown commodity'
        state = _pick_first(row, ['state_name', 'state', 'stateName']) or ''
        district = _pick_first(row, ['district_name', 'district', 'districtName']) or ''
        variety = _pick_first(row, ['variety_name', 'variety']) or ''
        report_date = _pick_first(row, ['date', 'arrival_date', 'report_date']) or ''
        arrival_qty = _pick_first(row, ['arrival_qty', 'arrival', 'quantity', 'qty']) or ''
        min_price = _pick_first(row, ['min_price', 'minimum_price', 'price_min']) or _pick_first(row, ['modal_price', 'price']) or ''
        max_price = _pick_first(row, ['max_price', 'maximum_price', 'price_max']) or _pick_first(row, ['modal_price', 'price']) or ''

        return {
            'status': 'success',
            'market': market,
            'commodity': commodity,
            'state': state,
            'district': district,
            'variety': variety,
            'report_date': report_date,
            'arrival_qty': arrival_qty,
            'min_price': _normalize_price_value(min_price),
            'max_price': _normalize_price_value(max_price),
        }

    if isinstance(row, (list, tuple)) and len(row) >= 8:
        return {
            'status': 'success',
            'market': str(row[3]).strip() if row[3] is not None else 'Unknown market',
            'commodity': str(row[4]).strip() if row[4] is not None else 'Unknown commodity',
            'min_price': _normalize_price_value(row[6]),
            'max_price': _normalize_price_value(row[7]),
        }

    return None


def get_real_prediction(image_path):
    try:
        safe_name = secure_filename(os.path.basename(image_path)) or 'upload.jpg'
        safe_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        if os.path.abspath(safe_path) != os.path.abspath(image_path):
            shutil.copy(image_path, safe_path)

        client = Client(HUGGING_FACE_URL)
        upload_path = safe_path if os.path.exists(safe_path) else image_path
        mime_type = mimetypes.guess_type(upload_path)[0] or 'image/jpeg'
        result = client.predict(
            {
                'path': upload_path,
                'url': None,
                'size': None,
                'orig_name': os.path.basename(upload_path),
                'mime_type': mime_type,
                'is_stream': False,
                'meta': {'_type': 'gradio.FileData'},
            },
            api_name='/predict',
        )

        confidences = None
        if isinstance(result, dict):
            confidences = result.get('confidences') or result.get('data') or result.get('predictions')
        elif isinstance(result, (list, tuple)) and result:
            confidences = result[0]

        if not confidences:
            return 'Error: Unexpected response from the model.'

        if confidences and isinstance(confidences[0], list):
            confidences = [{'label': item[0], 'confidence': item[1]} for item in confidences]
        elif confidences and isinstance(confidences[0], dict) and 'label' not in confidences[0]:
            confidences = [
                {
                    'label': item.get('class_name') or item.get('name') or 'Unknown',
                    'confidence': item.get('confidence') or item.get('score') or 0,
                }
                for item in confidences
            ]

        top_prediction = max(confidences, key=lambda item: item['confidence'])
        confidence_value = top_prediction['confidence']
        if confidence_value > 1:
            confidence_value = confidence_value / 100.0
        return f"Model Prediction: '{top_prediction['label']}' with {confidence_value:.2%} confidence."

    except Exception:
        app.logger.exception('Error calling Hugging Face API')
        return 'Error: Prediction service is temporarily unavailable. Please try again later.'


def fetch_market_prices(state_name='Karnataka', commodity_name='Orange'):
    filters = _load_agmarknet_filters()
    state_id = _find_id(filters.get('state_data', []), ['state_name', 'state'], state_name, ['state_id', 'id'])
    commodity_id = _find_id(filters.get('cmdt_data', []), ['cmdt_name', 'commodity_name', 'name'], commodity_name, ['cmdt_id', 'id'])

    if state_id is None:
        return [_error_row(f"'{state_name}' is not available in the live Agmarknet filters.")]
    if commodity_id is None:
        return [_error_row(f"'{commodity_name}' is not available in the live Agmarknet filters.")]

    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    payload = {
        'dashboard': 'statewise_price_arrival_sp',
        'states': [state_id],
        'commodities': [commodity_id],
        'from_date': from_date,
        'to_date': to_date,
        'format': 'json',
    }

    try:
        response = requests.post(
            f'{AGMARKNET_API_BASE}/dashboard-data/',
            json=payload,
            headers=REQUEST_HEADERS,
            timeout=20,
        )

        if response.status_code == 429:
            return [_error_row('Agmarknet is rate limiting live requests right now. Please retry in a minute.')]

        response.raise_for_status()
        data = response.json()

        if not data.get('status'):
            return [_error_row(data.get('message', 'Live market data is temporarily unavailable.'))]

        raw_records = []
        if isinstance(data.get('data'), dict):
            raw_records = data['data'].get('records') or data['data'].get('rows') or []
        elif isinstance(data.get('data'), list):
            raw_records = data['data']

        normalized_rows = []
        for record in raw_records:
            normalized = _normalize_report_row(record)
            if normalized:
                normalized_rows.append(normalized)

        if not normalized_rows:
            return [_error_row(f'No live price rows were returned for {commodity_name} in {state_name}.')]

        return normalized_rows[:10]
    except Exception as exc:
        app.logger.exception('Agmarknet live price fetch failed')
        return [_error_row(f'Live market data could not be loaded: {exc}')]


def parse_query_with_keywords(query):
    query = query.lower()
    state = 'Unknown'
    commodity = 'Unknown'
    if 'karnataka' in query:
        state = 'Karnataka'
    if 'maharashtra' in query:
        state = 'Maharashtra'
    if 'tur' in query or 'pigeon pea' in query:
        commodity = 'Pigeon Pea (Tur)'
    if 'onion' in query:
        commodity = 'Onion'
    if 'orange' in query:
        commodity = 'Orange'
    return {'state': state, 'commodity': commodity}


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = get_real_prediction(filepath)
        return render_template('result.html', prediction_text=prediction)
    return render_template('result.html', prediction_text='Error: No file selected.')


@app.route('/market-prices')
def market_prices():
    price_data = fetch_market_prices()
    return render_template('prices.html', price_data=price_data)


@app.route('/query-price', methods=['POST'])
def query_price():
    user_query = request.form.get('user_query', '')
    if not user_query:
        return render_template('result.html', prediction_text='Error: You did not enter a query.')
    entities = parse_query_with_keywords(user_query)
    state = entities.get('state', 'Unknown')
    commodity = entities.get('commodity', 'Unknown')
    price_data = fetch_market_prices(state_name=state, commodity_name=commodity)
    return render_template('prices.html', price_data=price_data)


@app.route('/ai_query', methods=['GET', 'POST'])
def ai_query():
    return render_template('ai_query.html')
