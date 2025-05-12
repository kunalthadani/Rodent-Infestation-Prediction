import mlflow.sklearn
import mlflow
import os


mlflow_username = os.getenv("MLFLOW_USERNAME")
mlflow_password = os.getenv("MLFLOW_PASSWORD")
mlflow_host = os.getenv("MLFLOW_HOST")
mlflow_port = os.getenv("MLFLOW_PORT")
mlflow_uri = "http://129.114.26.75:8000"


mlflow.set_tracking_uri(mlflow_uri)

import pandas as pd
import numpy as np
import requests
from mlflow.tracking import MlflowClient

import warnings
warnings.filterwarnings("ignore")

# boro = 'Brooklyn'
import sys


boro = sys.argv[1] if len(sys.argv) > 1 else "Brooklyn"
print(boro)

chunks = pd.read_csv('/mnt/rodent/data/all radius with weather no permits.csv', chunksize=500_000)
df = pd.concat(chunks)
df.drop(columns = ['rat_complaints_0.1mi',
 'rat_complaints_0.2mi',
 'rat_complaints_0.3mi',
 'rat_complaints_0.4mi','rat_complaints_0.6mi',
 'rat_complaints_0.7mi',
 'rat_complaints_0.8mi',
 'rat_complaints_0.9mi',
 'rat_complaints_1.0mi',], inplace = True)
df.drop(columns= ['temperature_2m_min', 'temperature_2m_max', 'precipitation_sum',
       'precip_day'], inplace = True)
df.dropna(subset = 'score', inplace = True)
violations = [i for i in df.columns if 'violation_code' in i]
df = df.groupby(['camis', 'month', 'rat_complaints_0.5mi', 'dba', 'latitude', 'longitude', 'boro']).sum(violations).reset_index()
df = df[df.boro == boro]
dba = df[['camis', 'dba']].drop_duplicates()
dba.to_csv(f"/mnt/rodent/inference_data/dba_" + boro + ".csv", index = False)
df['month'] = pd.to_datetime(df['month'])
health_data_agg = df.drop(columns = ['dba', 'latitude', 'longitude', 'boro'])
health_data_agg = health_data_agg[health_data_agg.month > '2000-01']
health_data_agg.to_csv(f"/mnt/rodent/inference_data/inf_feats_" + boro + ".csv", index = False)
health_data_agg = health_data_agg.sort_values(['camis','month'])
for lag in range(1, 4):
    health_data_agg[f'inspection_month_lag{lag}'] = (
        health_data_agg
          .groupby('camis')['month']
          .shift(lag)
    )


    health_data_agg[f'days_since_inspection_lag{lag}'] = (
        health_data_agg['month']
        - health_data_agg[f'inspection_month_lag{lag}']
    ).dt.days


    health_data_agg[f'score_lag{lag}'] = (
        health_data_agg.groupby('camis')['score']
          .shift(lag)
    )

    health_data_agg[f'rat_complaint_count{lag}'] = (
        health_data_agg.groupby('camis')['rat_complaints_0.5mi']
          .shift(lag)
    )


    for vc in violations:
          health_data_agg[f'{vc}_lag{lag}'] = (
              health_data_agg
                .groupby('camis')[vc]
                .shift(lag)
          )

health_data_agg.drop(columns = ['inspection_month_lag1', 'inspection_month_lag2', 'inspection_month_lag3'], inplace = True)

health_data_agg['pred'] = 0
health_data_agg.loc[(health_data_agg['violation_code_04L'] > 0) | (health_data_agg['violation_code_04K'] > 0), 'pred'] = 1
health_data_agg.drop(columns = violations, inplace = True)

health_data_agg.drop(columns = 'score', inplace = True)
# print(list(health_data_agg.columns))

"""***XGBoost***"""


mlflow.set_experiment(f"restaurant_infestation_predictor_test_" + boro)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import metrics

keys = health_data_agg['camis']
X = health_data_agg.drop(['camis','month',  'pred'], axis=1)
y = health_data_agg['pred']
health_data_agg.to_csv(f"/mnt/rodent/inference_data/test_feats_" + boro + ".csv", index = False)
# X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
#     X, y, keys,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

months = health_data_agg['month'].sort_values().unique()
test_months = months[-2:]

is_test  = health_data_agg['month'].isin(test_months)
is_train = ~is_test

X_train, y_train, keys_train = X.loc[is_train], y.loc[is_train], keys.loc[is_train]
X_test,  y_test,  keys_test  = X.loc[is_test],  y.loc[is_test],  keys.loc[is_test]


df_test_features = X_test.assign(key=keys_test)
df_test_features = df_test_features.assign(y = y_test)


objective = 'auc'
n_estimators=200
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
early_stopping_rounds=10

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric=objective,
    scale_pos_weight=scale_pos_weight,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=42,
    early_stopping_rounds=early_stopping_rounds,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

roc_auc = metrics.roc_auc_score(y_test, y_proba)

accuracy = metrics.accuracy_score(y_test, y_pred)

f1 = metrics.f1_score(y_test, y_pred)

recall = metrics.recall_score(y_test, y_pred)

precision = metrics.precision_score(y_test, y_pred)

df_results = X_test.copy()
df_results['key']       = keys_test
df_results['actual']    = y_test
df_results['predicted'] = y_pred
df_results['proba_1']   = y_proba

import numpy as np

from sklearn.calibration import calibration_curve
import pandas as pd

bins = np.arange(0, 1.1, 0.1)

fraction_of_positives, mean_predicted_value = calibration_curve(df_results['actual'], df_results['proba_1'],
                                                                n_bins=10, strategy='uniform')


df_results['binned'] = pd.cut(df_results['proba_1'], bins=bins, include_lowest=True)


df_binned = df_results.groupby('binned').size().reset_index(name='count')

agg_df = df_results.groupby('binned').agg(
    actual_sum=('actual', 'sum'),
    actual_count=('actual', 'count') 
).reset_index()

agg_df['ratio'] = agg_df.actual_sum / agg_df.actual_count 
agg_df['perfect'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

agg_df.dropna(inplace = True)

x = np.arange(len(agg_df))
y = agg_df['ratio'].values  

slope, intercept = np.polyfit(x, y, 1) 

print(f"Slope of the ratio column: {slope}")

with mlflow.start_run() as run:
    mlflow.log_param('eval_metric', 'auc')
    mlflow.log_param('scale_pos_weight', scale_pos_weight)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('subsample', subsample)
    mlflow.log_param('colsample_bytree', colsample_bytree)
    mlflow.log_param('early_stopping_rounds', early_stopping_rounds)
    mlflow.log_param('model_type', 'XGBoost')


    mlflow.log_metric('ROC AUC', roc_auc)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('Calibration slope', slope)

    # mlflow.sklearn.log_model(model, 'model')
    run_id = run.info.run_id
    if boro == 'Staten Island':
        boro = 'SI'
    mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="updated_model",           
                registered_model_name=f"restaurant-infestation-predictor-" +boro    
            )

