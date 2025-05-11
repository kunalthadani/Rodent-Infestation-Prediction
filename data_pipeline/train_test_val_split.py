import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()

processed_data_path = os.environ.get("TRANSFORMED_DATA_PATH")


# Train test val split will be done like this
# Validation data will be current month data
# Testing data will be the 3 previous month barring the current month
# Training data will be the remaining data

df = pd.read_csv(f"{processed_data_path}/all_radius.csv")

df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

# This will be current month
val_month = datetime.today().replace(day=1)

# Prev 3 months barring the current month
test_months = [val_month - relativedelta(months=i) for i in range(1, 4)]

val_mask = df['month'] == val_month
test_mask = df['month'].isin(test_months)
train_mask = ~(val_mask | test_mask)

# Split the data
val_df = df[val_mask]
test_df = df[test_mask]
train_df = df[train_mask]

train_df.to_csv(f"{processed_data_path}/train/all_radius.csv",header=True,index=False)
test_df.to_csv(f"{processed_data_path}/test/all_radiust.csv",header=True,index=False)
val_df.to_csv(f"{processed_data_path}/val/all_radius.csv",header=True,index=False)