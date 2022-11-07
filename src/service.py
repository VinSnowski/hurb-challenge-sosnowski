import pandas as pd
import bentoml
from bentoml.io import JSON
from src.features.build_features import FeatureBuilder

catboost_runner = bentoml.mlflow.get("catboost_model:latest").to_runner()

svc = bentoml.Service("catboost", runners=[catboost_runner])


@svc.api(input=JSON(), output=JSON())
async def predict(input_X: dict):

    X, y = FeatureBuilder.build_features(pd.json_normalize(input_X))
    result = catboost_runner.predict.async_run(X)
    return await result
