import os
import json
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# Read all data from .env file

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofnewyork.us",None, username="", password="", timeout=600)


def download_dataset(client, dataset, description, limit=2000):
    offset = 0
    final_data = []
    print(f"Downloading dataset: {dataset} - {description}")
    print("Downloading attachments...")
    client.download_attachments(dataset, download_dir="./dataset_attachments/")
    print("Downloading data...")
    all_data = []
    with open(f"{dataset} - {description}.json", "w") as f:
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
    print(f"Saved to file: {dataset} - {description}.json\nTotal records: {len(final_data)}\n\n")

datasets = {
    # "9nt8-h7nd": "Neighborhood Tabulation Data",
    # "cvf2-zn8s": "311 Rodent Complaints",
    # "43nn-pn8j": "NYC Restaurant Inspection Results",
    # "rv63-53db": "Garbage Collection Frequencies",
    # "ebb7-mvp5": "Monthly Tonnage Data",
    # "rbx6-tga4": "DOB NOW: Build - Approved Permits",
    # "erm2-nwe9": "311 Service Requests",
    # "3q43-55fe": "Rat Sightings",
    "ipu4-2q9a": "DOB Permit Issuance"
}

for _,dataset in enumerate(datasets):
    download_dataset(client,dataset,datasets[dataset])
