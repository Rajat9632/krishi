from flask import Flask, render_template

app = Flask(__name__)

# --- Simulated Data Function ---
def fetch_market_prices():
    fake_data = [
        {'market': 'Yadgir', 'commodity': 'Pigeon Pea (Tur)', 'min_price': '9800', 'max_price': '10200'},
        {'market': 'Bengaluru', 'commodity': 'Onion', 'min_price': '1500', 'max_price': '2000'}
    ]
    return fake_data

# --- Routes ---
@app.route('/')
def homepage():
    # For this test, we'll just link to the prices page.
    return '<a href="/market-prices">View Market Prices</a>'

@app.route('/market-prices')
def market_prices():
    price_data = fetch_market_prices()
    # Create a simple table string to display the data
    table = '<table border="1"><tr><th>Market</th><th>Commodity</th><th>Min Price</th><th>Max Price</th></tr>'
    for item in price_data:
        table += f"<tr><td>{item['market']}</td><td>{item['commodity']}</td><td>{item['min_price']}</td><td>{item['max_price']}</td></tr>"
    table += '</table>'
    return table