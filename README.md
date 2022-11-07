

# Machine Learning Engineer (Platform) Challenge for Hurb

## Containerized Bento Service (API) + MLFlow Tracking Integration serving a CatBoost classifier

### Setup and run

To start the application, go to the root directory and simply run

```bash
docker compose up
```

Two docker containers will be running, bentoml and mlflow. A model is generated at launch and is readily available for POST requests at [`http://localhost:3000/predict`](http://localhost:3000/predict) and the MLFlow server is located at [`http://localhost:5000/`](http://localhost:3000/predict)

### Usage

The API requires a JSON input with the following parameters, such as:

```json
{
"hotel": "Resort Hotel",
"lead_time": 96,
"arrival_date_year": 2015,
"arrival_date_month": "July",
"arrival_date_week_number": 27,
"arrival_date_day_of_month": 2,
"stays_in_weekend_nights": 2,
"stays_in_week_nights": 5,
"adults": 2,
"children": 0.0,
"babies": 0,
"meal": "BB",
"country": "ESP",
"market_segment": "Offline TA/TO",
"distribution_channel": "TA/TO",
"is_repeated_guest": 0,
"previous_cancellations": 0,
"previous_bookings_not_canceled": 0,
"reserved_room_type": "A",
"assigned_room_type": "A",
"booking_changes": 0,
"deposit_type": "No Deposit",
"agent": 134.0,
"company": 0,
"days_in_waiting_list": 0,
"customer_type": "Transient",
"adr": 58.95,
"required_car_parking_spaces": 0,
"total_of_special_requests": 1,
"reservation_status": "Check-Out",
"reservation_status_date": "2015-07-09"
}
```

And the response will also be a JSON with the following format:

```json
{
  "predicts_cancelling": true or false
}
```

An example of calling the API via bash with curl:

```bash
curl -v -H "Content-Type: application/json" -d '{
"hotel": "Resort Hotel",
"lead_time": 96,
"arrival_date_year": 2015,
"arrival_date_month": "July",
"arrival_date_week_number": 27,
"arrival_date_day_of_month": 2,
"stays_in_weekend_nights": 2,
"stays_in_week_nights": 5,
"adults": 2,
"children": 0.0,
"babies": 0,
"meal": "BB",
"country": "ESP",
"market_segment": "Offline TA/TO",
"distribution_channel": "TA/TO",
"is_repeated_guest": 0,
"previous_cancellations": 0,
"previous_bookings_not_canceled": 0,
"reserved_room_type": "A",
"assigned_room_type": "A",
"booking_changes": 0,
"deposit_type": "No Deposit",
"agent": 134.0,
"company": 0,
"days_in_waiting_list": 0,
"customer_type": "Transient",
"adr": 58.95,
"required_car_parking_spaces": 0,
"total_of_special_requests": 1,
"reservation_status": "Check-Out",
"reservation_status_date": "2015-07-09"
}' http://localhost:3000/predict
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── .env               <- Environment variables 
    │
    ├── Dockerfile_mlflow  <- Dockerfile to generate MLFlow docker image 
    │
    ├── Dockerfile_bentoml <- Dockerfile to generate BentoML docker image 
    │
    ├── commands.sh        <- Necessary commands to initiate BentoML image
    │
    ├── bentofile.yaml     <- Instructions to generate bento
    │
    ├── compose.yml        <- Docker compose file to run the application
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── service.py     <- BentoML service that runs predictions on demand
    |   |
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    └── test                <- Unit tests
        └── test_data.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
