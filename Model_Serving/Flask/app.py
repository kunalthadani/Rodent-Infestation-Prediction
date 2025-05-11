# app.py
import pandas as pd
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import sklearn
import logging
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import lightning as L

from prometheus_client import Gauge, Counter


mlflow.set_tracking_uri("http://129.114.26.75:8000")

MODEL_NAMES_XGBOOST = {
    "Manhattan":     "restaurant-infestation-predictor-Manhattan",
    "Brooklyn":      "restaurant-infestation-predictor-Brooklyn",
    "Queens":        "restaurant-infestation-predictor-Queens",
    "Bronx":         "restaurant-infestation-predictor-Bronx",
    "Staten Island": "restaurant-infestation-predictor-SI"
}

MODEL_NAMES_GRAPH = {
    "Manhattan":     "Graph_Manhattan",
    "Brooklyn":      "Graph_Brooklyn",
    "Queens":        "Graph_Queens",
    "Bronx":         "Graph_Bronx",
    "Staten Island": "Graph_Staten Island"
}

ENV_CONFIG = {
    "window_size": 3,
    "hidden_channels": 32,
    "out_channels": 1,
    "heads": 4,
    "lr": 5e-4,
    "batch_size": 4,
    "num_epochs": 20,
    "borough": "Brooklyn"
}

CUTOFF_MONTH = "2025-04"

app = Flask(__name__)
metrics = PrometheusMetrics(app)

class LightningGAT(L.LightningModule):
    def __init__(
        self, in_channels, hidden_channels,
        out_channels, heads, lr
    ):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.save_hyperparameters()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads=1, concat=False)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        out = F.softplus(x)
        return out.squeeze()


g_accuracy  = Gauge('model_accuracy',   'Model accuracy',   ['borough'])
g_f1        = Gauge('model_f1_score',   'Model F1 score',   ['borough'])
g_recall    = Gauge('model_recall',     'Model recall',     ['borough'])
g_precision = Gauge('model_precision',  'Model precision',  ['borough'])

predict_calls = Counter(
    'predict_requests_total',
    'Total number of /predict calls',
    ['borough']
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
app.logger.propagate = True

def get_highest_model_version(borough: str, model_type: str) -> int:
    client = MlflowClient()
    if model_type == "xgboost":
        name = MODEL_NAMES_XGBOOST[borough]
    else:
        name = MODEL_NAMES_GRAPH[borough]
    
    versions = client.search_model_versions(f"name='{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    app.logger.info("Latest version for %s is %s", borough, latest.version)
    return int(latest.version)

def get_xgboost_model(borough: str):
    version = get_highest_model_version(borough, "xgboost")
    uri = f"models:/{MODEL_NAMES_XGBOOST[borough]}/{version}"
    return mlflow.sklearn.load_model(uri)

def get_graph_model(borough: str):
    version = get_highest_model_version(borough, "graph")
    uri = f"models:/{MODEL_NAMES_GRAPH[borough]}/{version}"
    app.logger.info("uri = %s",uri)
    model = mlflow.pytorch.load_model(uri,map_location=torch.device('cpu'))
    sd = model.state_dict()
    if borough in ['Queens', 'Brooklyn','Manhattan']:
        ENV_CONFIG["hidden_channels"] = 16
    else:
        ENV_CONFIG["hidden_channels"] = 32
    app.logger.info("ENV_CONFIG['hidden_channels'] %s",ENV_CONFIG["hidden_channels"])
    model = LightningGAT(
    ENV_CONFIG['window_size'] + 12, ENV_CONFIG['hidden_channels'],
    ENV_CONFIG['out_channels'], ENV_CONFIG['heads'], ENV_CONFIG['lr']
    )
    model.load_state_dict(sd)
    return model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', boroughs=list(MODEL_NAMES_GRAPH))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    borough = data.get('borough')
    if borough not in MODEL_NAMES_GRAPH:
        return jsonify(error="Invalid borough"), 400

    TEST_SAMPLES_PATH   = f"/mnt/rodent/inference_data/test_samples_{borough.replace(' ', '_')}.pt"
    EDGE_INDEX_PATH     = f"/mnt/rodent/inference_data/edge_index_{borough.replace(' ', '_')}.pt"
    # OUTPUT_CSV_PATH     = f"/mnt/rodent/inference_data/predictions_by_month_{borough.replace(' ', '_')}.csv"

    test_samples = torch.load(TEST_SAMPLES_PATH, map_location=torch.device("cpu"), weights_only=False)
    edge_index = torch.load(EDGE_INDEX_PATH, map_location=torch.device("cpu"))

    model_graph  = get_graph_model(borough)
    rows = []
    for sample in test_samples:
        month  = sample['month']
        y_true = sample['y'].numpy()
        camis  = sample['camis']
        x      = sample['x'].to(torch.device("cpu"))
        data   = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            y_pred = model_graph(data).cpu().numpy()
        for c, a, p in zip(camis, y_true, y_pred):
            rows.append({
                'camis':     c,
                'month':     str(month),
                'actual':    float(a),
                'predicted': float(p),
            })

    op = pd.DataFrame(rows)
    app.logger.info("Length of OP is =%s", len(op))
    # CUTOFF_MONTH = op.month.max()
    op = op[op.month == str(CUTOFF_MONTH)]
    app.logger.info("Length of OP after is =%s", len(op))
    op.drop(columns = ['actual'], inplace = True)
    op.rename(columns = {'predicted': 'rat_complaints_0.5mi'}, inplace = True)
    health_data_agg = pd.read_csv(f"/mnt/rodent/inference_data/inf_feats_" + borough + ".csv")
    health_data_agg['month'] = pd.to_datetime(health_data_agg['month'])
    health_data_agg = health_data_agg[health_data_agg.month < CUTOFF_MONTH]
    app.logger.info("Length of health_data_agg is =%s", len(health_data_agg))
    health_data_agg = pd.concat([op,health_data_agg])
    health_data_agg = health_data_agg.reset_index(drop = True)
    health_data_agg['month'] = pd.to_datetime(health_data_agg['month'])
    violations = [i for i in health_data_agg.columns if 'violation_code' in i]
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
    health_data_agg.drop(columns = violations, inplace = True)

    health_data_agg.drop(columns = 'score', inplace = True)

    health_data_agg = health_data_agg[health_data_agg.month == CUTOFF_MONTH]
    app.logger.info("Length of health_data_agg 2 is =%s", len(health_data_agg))
    test_feats= pd.read_csv(f"/mnt/rodent/inference_data/test_feats_" + borough + ".csv")
    test_feats = test_feats[['camis','month', 'pred']]
    test_feats['month'] = pd.to_datetime(test_feats['month'])
    dba = pd.read_csv(f"/mnt/rodent/inference_data/dba_" + borough + ".csv")
    health_data_agg = health_data_agg.merge(test_feats, on = ['camis', 'month'], how = 'left').merge(dba, on = 'camis', how = 'left')
    app.logger.info("Length of health_data_agg 3 is =%s", len(health_data_agg))
    model = get_xgboost_model(borough)

    X = health_data_agg.drop(columns = ['camis', 'month', 'pred', 'dba'])
    app.logger.info("Length of X is =%s", len(X))

    # Predict
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = model.predict(X)
    app.logger.info("Length of ys is =%s, %s", len(y_proba), len(y_proba))
    df = health_data_agg.copy()
    df['predicted'] = y_pred
    df['proba_1']   = y_proba
    df.rename(columns = {'pred' : 'actual'}, inplace = True)
    df = df[['dba', 'month', 'actual', 'predicted', 'proba_1']].dropna(subset = 'actual')
    y_test = df.actual
    y_pred = df.predicted
    y_proba = df.proba_1

    # Compute metrics
    app.logger.info("Length of y_test is =%s", len(y_test))
    app.logger.info("Length of y_pred is =%s", len(y_pred))
    accuracy  = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1_score  = sklearn.metrics.f1_score(y_test, y_pred)
    recall    = sklearn.metrics.recall_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)

    # Set Prometheus gauges (tagged by borough)
    g_accuracy.labels(borough=borough).set(accuracy)
    g_f1.labels(borough=borough).set(f1_score)
    g_recall.labels(borough=borough).set(recall)
    g_precision.labels(borough=borough).set(precision)

    predict_calls.labels(borough=borough).inc()

    # Top‐10
    top10 = (
        df.nlargest(10, 'proba_1')
          [['dba', 'proba_1']]
          .to_dict(orient='records')
    )
    return jsonify(top10=top10)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
