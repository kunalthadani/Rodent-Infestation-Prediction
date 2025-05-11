import openmeteo_requests
import time
import requests_cache
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from retry_requests import retry

load_dotenv()

# Run this initially
# Data is till the current date
borough_map = {
    "brooklyn": "40.386642,-73.52115",
    "queens": "40.45694,-73.49786",
    "manhattan": "40.45694,-73.62482",
    "bronx": "40.52724,-73.47458",
    "staten island": "40.597538,-74.08771"
}

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
end_date = date.today().strftime('%Y-%m-%d')
output_path = os.environ.get["DOWNLOAD_DATA_PATH","."]

for borough in borough_map:
    lat, long = borough_map[borough].split(",")
    params = {
        "latitude": float(lat),
        "longitude": float(long),
        "start_date": "2010-01-01",
        "end_date": f"{end_date}",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"]
    }
    print(f"Downloading initial bulk data for {borough} - {lat},{long}")
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(4).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(5).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["snowfall_sum"] = daily_snowfall_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe.to_csv(f"{output_path}/weather_{borough}.csv", index=False, header=True)
    print(f"Saved to file {output_path}/weather_{borough}.csv")
    
    # To prevent rate limiting
    time.sleep(60)
