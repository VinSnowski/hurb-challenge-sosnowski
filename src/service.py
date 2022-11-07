import bentoml
import warnings
import pandas as pd
from bentoml.io import JSON
from src.features.build_features import FeatureBuilder

warnings.filterwarnings("ignore")

catboost_runner = bentoml.mlflow.get("catboost_model:latest").to_runner()

svc = bentoml.Service("catboost", runners=[catboost_runner])


@svc.api(input=JSON(), output=JSON())
def predict(input_X: dict):
    X, y = FeatureBuilder.build_features(pd.json_normalize(input_X))
    result = catboost_runner.predict.run(X)

    if result[0] == 0:
        return {"predicts_cancelling": False}
    else:
        return {"predicts_cancelling": True}
