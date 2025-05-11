import ijson
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from functools import reduce
from dotenv import load_dotenv

load_dotenv()

rows = []

download_data_path = os.environ.get("DOWNLOAD_DATA_PATH",".")
processed_data_path = os.environ.get("TRANSFORMED_DATA_PATH",".")

# Only for permit issuance data as using read_json does not read the JSON file properly
# So, doing it this way
def stream_json_to_dataframe(file_path, batch_size=10_000):
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        batch = []
        for i, item in enumerate(parser):
            batch.append(item)
            if len(batch) == batch_size:
                yield pd.DataFrame(batch)
                batch = []
        if batch:  # yield remaining
            yield pd.DataFrame(batch)

chunks = stream_json_to_dataframe(f"{download_data_path}/ipu4-2q9a - DOB Permit Issuance.json")
permits = pd.concat(chunks, ignore_index=True)
restaurants = pd.read_json(f'{download_data_path}/43nn-pn8j - NYC Restaurant Inspection Results.json')
rat_complaints = pd.read_csv(f'{download_data_path}/Rat_Sightings_20250428.csv')

# These are the cols going to be used for transforming the data
restaurant_cols = ['camis', 'dba', 'boro', 'inspection_date', 'violation_code', 'latitude', 'longitude', 'score']
rat_cols = ['Latitude', 'Longitude', 'Created Date']
permit_cols = ["gis_latitude", "gis_longitude", "job_start_date", "expiration_date"]

restaurants = pd.DataFrame(restaurants, columns=restaurant_cols)
rat_complaints = pd.DataFrame(rat_complaints, columns=rat_cols)
permits = permits[permit_cols]

restaurants = restaurants.dropna(subset=['camis','dba', 'latitude', 'longitude', 'inspection_date'])
rat_complaints = rat_complaints.dropna(subset=['Latitude', 'Longitude', 'Created Date'])

restaurants['inspection_date'] = pd.to_datetime(restaurants['inspection_date'],errors = 'coerce')
rat_complaints['complaint_date'] = pd.to_datetime(rat_complaints['Created Date'],errors = 'coerce')

restaurants['inspection_month'] = restaurants['inspection_date'].dt.to_period('M')
rat_complaints['complaint_month'] = rat_complaints['complaint_date'].dt.to_period('M')

# Convert lat/lon to radians for BallTree
def to_radians(df):
    return np.radians(df[['Latitude', 'Longitude']].values)

# Preprocess dates
restaurants['month'] = pd.to_datetime(restaurants['inspection_date']).dt.to_period('M')
rat_complaints['month'] = pd.to_datetime(rat_complaints['complaint_date']).dt.to_period('M')

# Get rat complaints by month
rat_by_month = {month: df for month, df in rat_complaints.groupby('month')}

# Build restaurant monthly timeline
first_inspection = restaurants.groupby('camis').agg(
    lat=('latitude', 'first'),
    long=('longitude', 'first'),
    start_month=('month', 'min')
).reset_index()

# Create full monthly date range for each restaurant
current_month = pd.to_datetime('today').to_period('M')
monthly_rows = []

for _, row in first_inspection.iterrows():
    months = pd.period_range(row['start_month'], current_month, freq='M')
    for month in months:
        monthly_rows.append({
            'camis': row['camis'],
            'lat': row['lat'],
            'long': row['long'],
            'month': month
        })

monthly_rows_df = pd.DataFrame(monthly_rows)

# Lat long intersection for rat complaints
EARTH_RADIUS_MILES = 3958.8

for i in range(1,11):
    RADIUS_MILES = float(i/10)
    RADIUS_RAD = RADIUS_MILES / EARTH_RADIUS_MILES
    print(f"Processing for radius {RADIUS_MILES}mi")
    def process_row(row):
        lat_rad, lon_rad = np.radians([row['lat'], row['long']])
        point = np.array([[lat_rad, lon_rad]])
        month = row['month']

        rats = rat_by_month.get(month, pd.DataFrame())
        count = 0
        if not rats.empty:
            tree = BallTree(to_radians(rats), metric='haversine')
            count = tree.query_radius(point, r=RADIUS_RAD, count_only=True)[0]

        return count

    # Allocate more as this runs for 6+hrs
    counts = Parallel(n_jobs=-1, prefer='threads')(
        delayed(process_row)(row) for _, row in monthly_rows_df.iterrows()
    )

    monthly_rows_df[f'rat_complaints_{RADIUS_MILES}mi'] = counts

print("Merging all radius")
radius_rat_df = monthly_rows_df[['camis', 'month', 'rat_complaints_0.1mi', 'rat_complaints_0.2mi', 'rat_complaints_0.3mi', 'rat_complaints_0.4mi', 'rat_complaints_0.5mi', 'rat_complaints_0.6mi', 'rat_complaints_0.7mi', 'rat_complaints_0.8mi', 'rat_complaints_0.9mi', 'rat_complaints_1.0mi']]

primary_cols = ['dba', 'latitude', 'longitude', 'boro', 'inspection_date', 'score', 'violation_code']
df_merged = radius_rat_df.merge(
    restaurants[['camis', 'month'] + primary_cols],
    on=['camis', 'month'],
    how='left'
)

missing = df_merged['inspection_date'].isna()

# Incase the data does not have inspection data, merge from the other rows
fallback_cols = ['dba', 'latitude', 'longitude', 'boro']
df_fallback = radius_rat_df[missing].merge(
    restaurants[['camis'] + fallback_cols].drop_duplicates('camis'),
    on='camis',
    how='left'
)

columns_to_fill = ['dba', 'latitude', 'longitude', 'boro']

for col in columns_to_fill:
    df_merged[col] = df_merged.groupby('camis')[col].ffill()

# Save the original column as we are going to one hot encode it
df_merged['violation_code_original'] = df_merged['violation_code']

radius_rat_df_encoced = pd.get_dummies(df_merged, columns=['violation_code'])

for col in radius_rat_df_encoded.columns:
    if radius_rat_df_encoded[col].dropna().isin([0, 1]).all():
        radius_rat_df_encoded[col] = radius_rat_df_encoded[col].astype(int)
        

# Lat long intersection for building counts
for i in range(1,6):
    BUFFER_RADIUS_M = (i/10) * 1609.34  # 0.1 mile in meters
    NYC_CRS = "EPSG:2263"
    permit_df["job_start_date"] = pd.to_datetime(permit_df["job_start_date"],errors='coerce')
    permit_df["expiration_date"] = pd.to_datetime(permit_df["expiration_date"],errors='coerce')
    rad_df["month"] = pd.to_datetime(rad_df["month"])
    
    permit_df = permit_df.dropna(subset=["job_start_date", "expiration_date"])
    
    
    permit_df["month"] = permit_df.apply(
        lambda row: pd.date_range(row["job_start_date"], row["expiration_date"], freq="MS"),
        axis=1
    )
    permit_df = permit_df.explode("month")
    permit_df["month"] = permit_df["month"].dt.to_period("M").dt.to_timestamp()
    
    
    permit_gdf = gpd.GeoDataFrame(
        permit_df,
        geometry=gpd.points_from_xy(permit_df["gis_longitude"], permit_df["gis_latitude"]),
        crs="EPSG:4326"
    ).to_crs(NYC_CRS)
    
    rad_gdf = gpd.GeoDataFrame(
        rad_df,
        geometry=gpd.points_from_xy(rad_df["longitude"], rad_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(NYC_CRS)
    
    def process_month(month):
        rad_month = rad_gdf[rad_gdf["month"] == month].copy()
        permit_month = permit_gdf[permit_gdf["month"] == month].copy()

        if rad_month.empty or permit_month.empty:
            return pd.DataFrame(columns=["camis", "month", "building_count"])

        
        rad_month["geometry"] = rad_month.geometry.buffer(BUFFER_RADIUS_M)

        
        joined = gpd.sjoin(rad_month, permit_month, how="inner", predicate="contains")
        

        
        result = (
            joined.groupby(["camis", "month_left","month_right"])
            .size()
            .reset_index(name=f"building_count_{i/10}mi")
        )

        return result
        
months = rad_gdf["month"].unique()

results = Parallel(n_jobs=-1,prefer='threads')(delayed(process_month)(m) for m in months)
building_permits_df = pd.concat(results, ignore_index=True)
building_permits_df['month'] = newbuilding_permits_df_df['month_left']
building_permits_df = building_permits_df.drop('month_right',axis=1)
newbuilding_permits_df_df = building_permits_df.drop('month_left',axis=1)
building_permits_df["month"] = pd.to_datetime(building_permits_df["month"]).dt.to_period("M")


radius_rat_permit_df = pd.merge(radius_rat_df_encoced, building_permits_df, on=["camis", "month"], how="left")


boroughs = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']

weather_monthly_dfs = []

for boro in boroughs:
    bf = boro.lower()
    file_path = f'{download_data_path}/weather_{bf}.csv'
    weather_df = pd.read_csv(file_path)
    
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df['month'] = weather_df['date'].dt.to_period('M').astype(str)
    
    weather_df['precip_day'] = weather_df['precipitation_sum'] > 0
    
    monthly_agg = weather_df.groupby('month').agg({
        'temperature_2m_min': 'min',
        'temperature_2m_max': 'max',
        'precipitation_sum': 'sum',
        'precip_day': 'sum'
    }).reset_index()

    monthly_agg['boro'] = boro
    
    weather_monthly_dfs.append(monthly_agg)

# Combine all
all_weather_monthly = pd.concat(weather_monthly_dfs, ignore_index=True)

# Combine weather with radius
radius_rat_permit_weather_df = radius_rat_permit_df.merge(all_weather_monthly, on=['boro', 'month'], how='left')

print(f"Saving the file as {TRANSFORMED_DATA_PATH}/all_radius.csv")
radius_rat_permit_weather_df.to_csv(f"{TRANSFORMED_DATA_PATH}/all_radius.csv",header=True,index=False)