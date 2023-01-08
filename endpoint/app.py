from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from typing import Any
from sklearn.ensemble import RandomForestClassifier
import mlflow.pyfunc

classifier = RandomForestClassifier()
app = Flask(__name__)

def init_classifier_from_file():
     base_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
     data_file = os.path.join(base_dir, 'data/classifier.pkl') 
     return joblib.load(data_file)

def init_classifier_from_mlflow():
     model_name = "titanic-sk-learn-random-forest-reg-model"
     model_version = 1
     return mlflow.pyfunc.load_model(
          model_uri=f"models:/{model_name}/{model_version}"
     )

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = classifier.predict(query)
     return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
     #classifier = init_classifier_from_file()
     classifier = init_classifier_from_mlflow()
     app.run(host="0.0.0.0",port=8080, debug=True)