import os
import json
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# Read all data from .env file

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofnewyork.us",os.environ.get("NYC_OPENDATA_APP_TOKEN",None), username=os.environ.get("NYC_OPENDATA_USERNAME",""), password=os.environ.get("NYC_OPENDATA_PASSWORD",""), timeout=600)

output_path = os.environ.get("DOWNLOAD_DATA_PATH",".")

def download_dataset(client, dataset, description, limit=2000):
    offset = 0
    final_data = []
    print(f"Downloading dataset: {dataset} - {description}")
    print("Downloading attachments...")
    client.download_attachments(dataset, download_dir="{output_path}/dataset_attachments/")
    print("Downloading data...")
    all_data = []
    with open(f"{output_path}/{dataset} - {description}.json", "w") as f:
        while True:
            try:
                results = client.get(dataset, limit=limit, offset=offset)
            except Exception as e:
                print("Error downloading data")
                print(e)
                return
            final_data.extend(results)
            offset += limit
            if len(results) < limit:
                break
            all_data.extend(results)
        json.dump(all_data, f, indent=4)
    print(f"Saved to file: {output_path}/{dataset} - {description}.json\nTotal records: {len(final_data)}\n\n")

datasets = {
    "43nn-pn8j": "NYC Restaurant Inspection Results",
    "3q43-55fe": "Rat Sightings",
    "ipu4-2q9a": "DOB Permit Issuance"
}

for _,dataset in enumerate(datasets):
    download_dataset(client,dataset,datasets[dataset])
