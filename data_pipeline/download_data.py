#!/usr/bin/env python

# make sure to install these packages before running:
# pip install pandas
# pip install sodapy

import json
import pandas as pd
from sodapy import Socrata

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofnewyork.us","auth_key", username="", password="", timeout=600)


def download_dataset(client, dataset, description, limit=2000):
    offset = 0
    final_data = []
    print(f"Downloading dataset: {dataset} - {description}")
    print("Downloading attachments...")
    client.download_attachments(dataset, download_dir="./dataset_attachments/")
    print("Downloading data...")
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
            # if offset >= 5000:
            #     break
            json.dump(results, f, indent=4)
            f.write("\n")
    print(f"Saved to file: {dataset} - {description}.json\nTotal records: {len(final_data)}\n\n")

datasets = {
    # "9nt8-h7nd": "Neighborhood Tabulation Data",
    "cvf2-zn8s": "311 Rodent Complaints",
    # "43nn-pn8j": "NYC Restaurant Inspection Results",
    # "rv63-53db": "Garbage Collection Frequencies",
    # "ebb7-mvp5": "Monthly Tonnage Data",
    # "rbx6-tga4": "DOB NOW: Build – Approved Permits",
    # "erm2-nwe9": "311 Service Requests",
    "3q43-55fe": "Rat Sightings"
}

for _,dataset in enumerate(datasets):
    download_dataset(client,dataset,datasets[dataset])



# Example authenticated client (needed for non-public datasets):
# client = Socrata(data.cityofnewyork.us,
#                  MyAppToken,
#                  username="user@example.com",
#                  password="AFakePassword")

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.

# Neighborhood Tabulation Data
# client.download_attachments("9nt8-h7nd", download_dir="./")
# results = client.get("9nt8-h7nd", limit=2000)
# with open("./9nt8-h7nd.json", "w") as f:
#     json.dump(results, f, indent=4)


# 311 Rodent Complaints
# client = Socrata("data.cityofnewyork.us", None)
# dataset = "cvf2-zn8s"
# client.download_attachments(dataset, download_dir="./")
# results = client.get(dataset, limit=1000)
# print(results)
# offset = 0
# limit = 1000
# final_data = []
# with open(f"{dataset}.json", "w") as f:
#     while True:
#         results = client.get(dataset, limit=1000, offset=offset)
#         final_data.extend(results)
#         print(len(final_data))
#         offset += limit
#         # if len(results) < 1000:
#         #     break
#         if offset >= 5000:
#             break
#     json.dump(final_data, f, indent=4)



# NYC Restaurant Inspection Results
# dataset = "43nn-pn8j"
# client.download_attachments(dataset, download_dir="./")
# offset = 0
# limit = 1000
# final_data = []
# with open(f"{dataset}.json", "w") as f:
#     while True:
#         results = client.get(dataset, limit=1000, offset=offset)
#         final_data.extend(results)
#         print(len(final_data))
#         offset += limit
#         if offset >= 5000:
#             break
#     json.dump(final_data, f, indent=4)


# Garbage collection frequencies
# dataset = "rv63-53db"
# client.download_attachments(dataset, download_dir="./")
# offset = 0
# limit = 1000
# final_data = []
# with open(f"{dataset}.json", "w") as f:
#     while True:
#         results = client.get(dataset, limit=1000, offset=offset)
#         final_data.extend(results)
#         offset += limit
#         if len(results) < 1000:
#             break
#     json.dump(final_data, f, indent=4)


# Monthly tonnage data
# dataset = "ebb7-mvp5"
# client.download_attachments(dataset, download_dir="./")
# offset = 0
# limit = 1000
# final_data = []
# with open(f"{dataset}.json", "w") as f:
#     while True:
#         results = client.get(dataset, limit=1000, offset=offset)
#         final_data.extend(results)
#         offset += limit
#         print(len(final_data))
#         if len(results) < 999:
#             break
#         if offset >= 5000:
#             break
#     json.dump(final_data, f, indent=4)




# # Convert to pandas DataFrame
# results_df = pd.DataFrame.from_records(results)

# print(results_df)