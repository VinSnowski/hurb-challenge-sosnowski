#!/bin/bash
python src/models/train_model.py
python test/test_data.py
bentoml serve