# app.py
import pandas as pd
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import sklearn
import logging

from prometheus_client import Gauge, Counter

app = Flask(__name__)
metrics = PrometheusMetrics(app)

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

mlflow.set_tracking_uri("http://129.114.26.75:8000")  

MODEL_NAMES = {
    "Manhattan":     "restaurant-infestation-predictor-Manhattan",
    "Brooklyn":      "restaurant-infestation-predictor-Brooklyn",
    "Queens":        "restaurant-infestation-predictor-Queens",
    "Bronx":         "restaurant-infestation-predictor-Bronx",
    "Staten Island": "restaurant-infestation-predictor-SI"
}

def get_highest_model_version(borough: str) -> int:
    client = MlflowClient()
    name = MODEL_NAMES[borough]
    versions = client.search_model_versions(f"name='{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    app.logger.info("Latest version for %s is %s", borough, latest.version)
    return int(latest.version)

def get_model(borough: str):
    version = get_highest_model_version(borough)
    uri = f"models:/{MODEL_NAMES[borough]}/{version}"
    return mlflow.sklearn.load_model(uri)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', boroughs=list(MODEL_NAMES))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    borough = data.get('borough')
    if borough not in MODEL_NAMES:
        return jsonify(error="Invalid borough"), 400

    model = get_model(borough)
    fname = f"test_feats_{borough.replace(' ', '_')}.csv"
    df = pd.read_csv(fname)
    y_test = df.y
    X = df.drop(columns=['key', 'y'])

    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = model.predict(X)
    df['proba_1'] = y_proba

    accuracy  = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1_score  = sklearn.metrics.f1_score(y_test, y_pred)
    recall    = sklearn.metrics.recall_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)

    g_accuracy.labels(borough=borough).set(accuracy)
    g_f1.labels(borough=borough).set(f1_score)
    g_recall.labels(borough=borough).set(recall)
    g_precision.labels(borough=borough).set(precision)

    predict_calls.labels(borough=borough).inc()

    top10 = (
        df.nlargest(10, 'proba_1')
          [['key', 'proba_1']]
          .to_dict(orient='records')
    )
    return jsonify(top10=top10)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
