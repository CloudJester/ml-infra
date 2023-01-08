from flask import Flask, request, jsonify
from train.titanic.train import train_titanic
import sys
import joblib
import logging


import mlflow
import mlflow.sklearn


def basic_pickle_file():
    logging.info('Training Model...')
    classifler = train_titanic()
    logging.info('Exporting Model...')
    joblib.dump(classifler, 'endpoint/data/classifier.pkl')

def ml_flow():
    with mlflow.start_run(run_name="titanic_run") as run:
        #params = {"n_estimators": 5, "random_state": 42}
        logging.info('Training Model...')
        classifler = train_titanic()

        # Log parameters and metrics using the MLflow APIs
        #mlflow.log_params(params)
        #mlflow.log_param("param_1", randint(0, 100))
        #mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

        # Log the sklearn model and register as version 1
        logging.info('Exporting Model...')
        mlflow.sklearn.log_model(
            sk_model=classifler,
            artifact_path="titanic-sk-learn",
            registered_model_name="titanic-sk-learn-random-forest-reg-model"
        )

def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    ml_flow()
    #basic_pickle_file()
    logging.info('Done')


if __name__ == '__main__':
     main()