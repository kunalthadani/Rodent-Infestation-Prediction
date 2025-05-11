import openmeteo_requests
import time

import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta


# Run this everyday

borough_map = {
    "brooklyn": "40.386642,-73.52115",
    "queens": "40.45694,-73.49786",
    "manhattan": "40.45694,-73.62482",
    "bronx": "40.52724,-73.47458",
    "staten island": "40.597538,-74.08771"
}

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://api.open-meteo.com/v1/forecast"


for borough in borough_map:
    lat, long = borough_map[borough].split(",")
    csv_file = f"weather_{borough}.csv"
    borough_df = pd.read_csv(csv_file, parse_dates=['date'])
    last_date = borough_df.iloc[-1]['date'].date()
    yesterday = datetime.today().date() - timedelta(days=1)
    new_rows = []

    print(f"Downloading data for {borough}")
    print(f"Last date data we have {last_date}")
    next_date = last_date + timedelta(days=1)
    while next_date <= yesterday:
        print(f"Fetching data for {next_date}")

        params = {
            "latitude": float(lat),
            "longitude": float(long),
            "daily": ["rain_sum", "precipitation_hours", "snowfall_sum", "precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
            "start_date": next_date.strftime("%Y-%m-%d"),
            "end_date": next_date.strftime("%Y-%m-%d")
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        daily_rain_sum = daily.Variables(0).ValuesAsNumpy()
        daily_precipitation_hours = daily.Variables(1).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(2).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(4).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(5).ValuesAsNumpy()

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
        daily_dataframe.to_csv(csv_file, mode='a', header=False, index=False)
        next_date += timedelta(days=1)
        time.sleep(1)
    print("Saved to file")

    # To prevent rate limiting
    time.sleep(10)
