import pandas as pd
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import sklearn
import logging

mlflow.set_tracking_uri("http://129.114.26.75:8000")  # if needed

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# — your existing model names —
MODEL_NAMES = {
    "Manhattan":       "restaurant-infestation-predictor-Manhattan",
    "Brooklyn":        "restaurant-infestation-predictor-Brooklyn",
    "Queens":          "restaurant-infestation-predictor-Queens",
    "Bronx":           "restaurant-infestation-predictor-Bronx",
    "Staten Island":   "restaurant-infestation-predictor-SI"
}

# — logging setup (unchanged) —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
app.logger.propagate = True

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

# — your original HTML route, unchanged —
@app.route('/', methods=['GET'])
def index():
    # you can still handle GET/POST here if you want,
    # but POST won't actually run predictions anymore
    return render_template('index.html', boroughs=list(MODEL_NAMES))

# — new JSON API for prediction —
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    borough = data.get('borough')
    if borough not in MODEL_NAMES:
        return jsonify(error="Invalid borough"), 400

    # load the right CSV
    model = get_model(borough)
    fname = f"test_feats_{borough.replace(' ', '_')}.csv"
    df = pd.read_csv(fname)
    y_test = df.y
    X = df.drop(columns=['key', 'y'])

    # run your MLflow model

    proba = model.predict_proba(X)[:, 1]
    y_pred  = model.predict(X)
    df['proba_1'] = proba

    # roc_auc = metrics.roc_auc_score(y_test, y_proba)

    # Accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    # F1 Score
    f1 = sklearn.metrics.f1_score(y_test, y_pred)

    # Recall
    recall = sklearn.metrics.recall_score(y_test, y_pred)

    # Precision
    precision = sklearn.metrics.precision_score(y_test, y_pred)

    # grab top 10 and return as JSON
    top10 = (
        df.nlargest(10, 'proba_1')
          [['key', 'proba_1']]
          .to_dict(orient='records')
    )
    return jsonify(top10=top10)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
