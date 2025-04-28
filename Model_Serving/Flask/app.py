import joblib
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template

import os

app = Flask(__name__)

york_boroughs = [
    'Manhattan',
    'Brooklyn',
    'Queens',
    'Bronx',
    'Staten Island'
]

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        borough = request.form.get('borough')
        # TODO: replace this with your custom logic
        brf = joblib.load('brf_model.joblib')
        df_test_features = pd.read_csv('test_feats.csv')
        keys_test = df_test_features.key
        y_test = df_test_features.y
        X_test = df_test_features.drop(columns = ['key', 'y'])
        y_proba = brf.predict_proba(X_test)[:, 1]
        y_pred  = brf.predict(X_test)


        df_results = X_test.copy()
        df_results['key']       = keys_test
        df_results['actual']    = y_test
        df_results['predicted'] = y_pred
        df_results['proba_1']   = y_proba

        op = list(df_results.sort_values('proba_1', ascending = False).head(10).key)
        result = f"You selected: {borough}!"
    return render_template('index.html', boroughs=york_boroughs, result=result, op = op)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)