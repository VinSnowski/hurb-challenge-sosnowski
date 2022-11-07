import mlflow
import bentoml
import warnings
import logging
import os, dotenv
import pandas as pd
from numpy.random import seed
from catboost import CatBoostClassifier
from src.features.build_features import FeatureBuilder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

warnings.filterwarnings("ignore")
dotenv.load_dotenv(".env")


class CatBooostModel:
    def __init__(self):

        logging.info("Generating CatBoost model")
        seed(1000)
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.conf = None
        self.classifier = None
        self.logged_model = None
        self.bento_model = None

    def generate_model(self) -> None:

        filename = os.environ["RAW_DATA_FILENAME"]
        self.X_train, self.X_test, self.y_train, self.y_test = FeatureBuilder.full_pipeline(filename)

        self.classifier = CatBoostClassifier(iterations=100)
        self.classifier.fit(self.X_train, self.y_train)

        y_pred = self.classifier.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.conf = confusion_matrix(self.y_test, y_pred)

        logging.info(f"Model generated with accuracy: {self.accuracy}, recall: {self.recall} and precision: {self.precision}")

    def log_to_mlflow(self) -> None:

        MLFLOW_SERVER_URL = os.environ["MLFLOW_SERVER_URL"]
        mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
        mlflow.set_experiment("hotel-bookings")

        with mlflow.start_run():

            t_n, f_p, f_n, t_p = self.conf.ravel()

            mlflow.log_metric("TrueNeg", t_n)
            mlflow.log_metric("FalsePos", f_p)
            mlflow.log_metric("FalseNeg", f_n)
            mlflow.log_metric("TruePos", t_p)
            mlflow.log_metric("AccuracyScore", self.accuracy)
            mlflow.log_metric("RecallScore", self.recall)
            mlflow.log_metric("PrecisionScore", self.precision)

            self.logged_model = mlflow.catboost.log_model(self.classifier, "catboost")

        logging.info("Model logged to MLFlow: %s " % self.logged_model.model_uri)

    def log_to_bentoml(self, logged_model: mlflow.models.model.ModelInfo) -> None:

        # Import logged mlflow model to BentoML model store for serving:
        self.bento_model = bentoml.mlflow.import_model("catboost_model", logged_model.model_uri)
        logging.info("Model imported to BentoML: %s" % self.bento_model)

    def run_setup(self) -> None:

        self.generate_model()
        self.log_to_mlflow()
        self.log_to_bentoml(self.logged_model)


if __name__ == "__main__":
    model = CatBooostModel().run_setup()
