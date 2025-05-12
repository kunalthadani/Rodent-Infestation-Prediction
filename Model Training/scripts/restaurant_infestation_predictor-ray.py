
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, recall_score, precision_score
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback
import mlflow


URI      = "http://129.114.26.75:8000"
EXPERIMENT = "restaurant_infestation_predictor_test_Brooklyn_Ray_tune"

def load_and_prepare():
    
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
    df = df[df.boro == 'Brooklyn']
    dba = df[['camis', 'dba']].drop_duplicates()
    # dba.to_csv(f"/mnt/rodent/inference_data/dba_" + boro + ".csv", index = False)
    df['month'] = pd.to_datetime(df['month'])
    health_data_agg = df.drop(columns = ['dba', 'latitude', 'longitude', 'boro'])
    health_data_agg = health_data_agg[health_data_agg.month > '2000-01']
    # health_data_agg.to_csv(f"/mnt/rodent/inference_data/inf_feats_" + boro + ".csv", index = False)
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
    
    keys = health_data_agg['camis']
    X = health_data_agg.drop(['camis','month',  'pred'], axis=1)
    y = health_data_agg['pred']
    months = health_data_agg['month'].sort_values().unique()
    test_months = months[-2:]
    is_test  = health_data_agg['month'].isin(test_months)
    # health_data_agg.to_csv(f"/mnt/rodent/inference_data/test_feats_" + boro + ".csv", index = False)
    return X, y, is_test, keys



def train_xgb(config):
    mlflow.set_experiment(f"restaurant_infestation_predictor_test_Brooklyn")
    
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn import metrics

    
    X, y, is_test, keys = load_and_prepare()
    is_train = ~is_test
    # _X, _y = load_and_prepare()
    imbalance_ratio = (y==0).sum() / (y==1).sum()
    X_train, y_train, keys_train = X.loc[is_train], y.loc[is_train], keys.loc[is_train]
    X_test,  y_test,  keys_test  = X.loc[is_test],  y.loc[is_test],  keys.loc[is_test]

    df_test_features = X_test.assign(key=keys_test)
    df_test_features = df_test_features.assign(y = y_test)
    config["scale_pos_weight"] = imbalance_ratio

    model = xgb.XGBClassifier(**config)
    model.fit(X_train, y_train, eval_set=[(X_test,y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred  = model.predict(X_test)
    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }
    tune.report(metrics)

search_space = {
    "objective":             "binary:logistic",
    "use_label_encoder":     False,
    "eval_metric":           "auc",
    "n_estimators":          tune.choice([100, 200, 300]),
    "learning_rate":         tune.loguniform(1e-3, 1e-1),
    "max_depth":             tune.choice([4, 6, 8]),
    "subsample":             tune.uniform(0.6, 1.0),
    "colsample_bytree":      tune.uniform(0.6, 1.0),
    "early_stopping_rounds": tune.choice([5, 10, 20]),
    "random_state":          42,
}

scheduler = ASHAScheduler(metric="roc_auc", mode="max", grace_period=5, reduction_factor=2)
reporter  = CLIReporter(
    parameter_columns=["n_estimators","max_depth","learning_rate"],
    metric_columns=["roc_auc","accuracy","training_iteration"]
)

analysis = tune.run(
    train_xgb,
    resources_per_trial={"cpu": 8},
    config=search_space,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    callbacks=[
      MLflowLoggerCallback(
        tracking_uri=URI,
        experiment_name=EXPERIMENT
      )
    ],
    storage_path="s3://ray"
)

print("Best hyperparameters:", analysis.get_best_config(metric="roc_auc", mode="max"))
best_config = analysis.get_best_config(metric="roc_auc", mode="max")
print("Best config:", best_config)


X, y, is_test, keys = load_and_prepare()  
is_train = ~is_test
X_train, y_train, keys_train = X.loc[is_train], y.loc[is_train], keys.loc[is_train]
X_test,  y_test,  keys_test  = X.loc[is_test],  y.loc[is_test],  keys.loc[is_test]

neg = (y_train==0).sum()
pos = (y_train==1).sum()
best_config["scale_pos_weight"] = neg/pos

best_model = xgb.XGBClassifier(**best_config)
best_model.fit(X_train, y_train, eval_set=[(X_test,y_test)], verbose=False)

# mlflow.set_tracking_uri(f"http://{USERNAME}:{PASSWORD}@{HOST}:{PORT}")
# mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="best_xgb_model"):

    mlflow.log_params(best_config)

    y_proba = best_model.predict_proba(X_test)[:,1]
    y_pred  = best_model.predict(X_test)
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_proba))
    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1",        f1_score(y_test, y_pred))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))

    mlflow.sklearn.log_model(best_model, "best_model_2")
