from datetime import datetime

from app import fetch_market_prices


if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    price_data = fetch_market_prices()

    print(f'Fetching live price data as of {today}...')
    if price_data and isinstance(price_data[0], dict) and price_data[0].get('status') == 'error':
        print(price_data[0]['message'])
    else:
        for item in price_data:
            print(
                f"Market: {item.get('market', 'Unknown')}, "
                f"Commodity: {item.get('commodity', 'Unknown')}, "
                f"Min Price: {item.get('min_price', '')}, "
                f"Max Price: {item.get('max_price', '')}"
            )