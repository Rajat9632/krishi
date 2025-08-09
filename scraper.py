import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def fetch_market_prices(date_str, commodity_code="103", state_code="KK", market_code="23"):
    """
    Fetches agricultural price data for a specific date from Agmarknet.
    Now more flexible, with default values for Onion in Bengaluru.
    """
    # Base URL without query parameters
    base_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    
    # SUGGESTION: Parameters are now part of a dictionary for clarity
    params = {
        'Tx_Commodity': commodity_code,
        'Tx_State': state_code,
        'Tx_Market': market_code,
        'DateFrom': date_str,
        'DateTo': date_str,
        'Fr_Date': date_str,
        'To_Date': date_str,
        'Tx_Trend': '0',
        'Tx_CommodityHead': 'Onion',      # Note: These head values may need to be updated
        'Tx_StateHead': 'Karnataka',    # if the commodity/state changes. For a more
        'Tx_DistrictHead': 'Bangalore', # advanced script, these would also be dynamic.
        'Tx_MarketHead': 'Bangalore'
    }

    # SUGGESTION: Add a User-Agent header to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Fetching data for {date_str}...")

    try:
        # Pass headers and params to the request
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        data_table = soup.find('table', id='cphBody_GridPriceData')

        if not data_table:
            return ["Price data table not found on the page."]

        rows = data_table.find_all('tr')[1:]

        if not rows:
            return ["No data rows found in the table."]
            
        # SUGGESTION: Return a list of dictionaries for structured data
        results = []
        for row in rows:
            columns = row.find_all('td')
            data = [col.text.strip() for col in columns]

            if len(data) > 0 and "No Data Found" in data[0]:
                return ["'No Data Found' message was displayed on the site."]
            
            # Check if there are enough columns before trying to access them
            if len(data) >= 8:
                try:
                    # SUGGESTION: Convert prices to numbers for future use
                    market_data = {
                        "market": data[3],
                        "min_price": int(data[6]),
                        "max_price": int(data[7])
                    }
                    results.append(market_data)
                except (ValueError, IndexError) as e:
                    # Handle cases where price isn't a number or index is wrong
                    print(f"Could not parse row: {data}. Error: {e}")
        
        if not results:
             return ["Data rows were found, but none could be parsed successfully."]
             
        return results

    except requests.exceptions.RequestException as e:
        return [f"An error occurred: {e}"]


# --- Main script execution ---
if __name__ == "__main__":
    # SUGGESTION: Dynamically get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    target_date = yesterday.strftime("%d-%b-%Y") # Format: 30-Jul-2025

    # We can still call it with the defaults for Onion/Bengaluru
    price_data = fetch_market_prices(target_date)

    print("\n--- Scraped Data ---")
    if isinstance(price_data, list) and len(price_data) > 0:
        # Check if the first item is a dictionary (successful parse) or a string (error message)
        if isinstance(price_data[0], dict):
            for item in price_data:
                # Now we can format it nicely and even do calculations
                print(f"Market: {item['market']}, Min Price: ₹{item['min_price']:,}, Max Price: ₹{item['max_price']:,}")
        else:
             # Print error messages
             for item in price_data:
                print(item)