import logging
import warnings
import os, dotenv
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv(".env")


class FeatureBuilder:
    @staticmethod
    def read_data(filename: str) -> pd.DataFrame:
        try:
            raw_data_path = os.environ["RAW_DATA_PATH"]
            df = pd.read_csv(raw_data_path + filename)
            return df
        except:
            raise Exception(f"Could not read {filename}")

    @staticmethod
    def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        # dropping columns that are not useful
        useless_col = [
            "days_in_waiting_list",
            "arrival_date_year",
            "arrival_date_year",
            "assigned_room_type",
            "booking_changes",
            "reservation_status",
            "country",
            "days_in_waiting_list",
        ]
        df = df.drop(useless_col, axis=1)

        # creating numerical and categorical dataframes
        cat_cols = [col for col in df.columns if df[col].dtype == "O"]
        cat_df = df[cat_cols]
        cat_df["reservation_status_date"] = pd.to_datetime(cat_df["reservation_status_date"])
        cat_df["year"] = cat_df["reservation_status_date"].dt.year
        cat_df["month"] = cat_df["reservation_status_date"].dt.month
        cat_df["day"] = cat_df["reservation_status_date"].dt.day
        cat_df = cat_df.drop(["reservation_status_date", "arrival_date_month"], axis=1)

        # encoding categorical variables
        cat_df["hotel"] = cat_df["hotel"].map({"Resort Hotel": 0, "City Hotel": 1})
        cat_df["meal"] = cat_df["meal"].map({"BB": 0, "FB": 1, "HB": 2, "SC": 3, "Undefined": 4})
        cat_df["market_segment"] = cat_df["market_segment"].map(
            {
                "Direct": 0,
                "Corporate": 1,
                "Online TA": 2,
                "Offline TA/TO": 3,
                "Complementary": 4,
                "Groups": 5,
                "Undefined": 6,
                "Aviation": 7,
            }
        )
        cat_df["distribution_channel"] = cat_df["distribution_channel"].map({"Direct": 0, "Corporate": 1, "TA/TO": 2, "Undefined": 3, "GDS": 4})
        cat_df["reserved_room_type"] = cat_df["reserved_room_type"].map({"C": 0, "A": 1, "D": 2, "E": 3, "G": 4, "F": 5, "H": 6, "L": 7, "B": 8})
        cat_df["deposit_type"] = cat_df["deposit_type"].map({"No Deposit": 0, "Refundable": 1, "Non Refund": 3})
        cat_df["customer_type"] = cat_df["customer_type"].map({"Transient": 0, "Contract": 1, "Transient-Party": 2, "Group": 3})
        cat_df["year"] = cat_df["year"].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})

        # normalizing numerical variables
        num_df = df.drop(columns=cat_cols, axis=1)

        num_df["lead_time"] = np.log(num_df["lead_time"] + 1)
        num_df["arrival_date_week_number"] = np.log(num_df["arrival_date_week_number"] + 1)
        num_df["arrival_date_day_of_month"] = np.log(num_df["arrival_date_day_of_month"] + 1)
        num_df["agent"] = np.log(num_df["agent"] + 1)
        num_df["company"] = np.log(num_df["company"] + 1)
        num_df["adr"] = np.log(num_df["adr"] + 1)
        num_df["adr"] = num_df["adr"].fillna(value=num_df["adr"].mean())

        X = pd.concat([cat_df, num_df], axis=1)

        if len(df) == 1:
            y = pd.Series()
        else:
            X = X.drop("is_canceled", axis=1)
            y = df["is_canceled"]
        return X, y

    @staticmethod
    def get_train_test_sets(X: pd.DataFrame, y: pd.Series, save_processed_data=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        # splitting data into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

        if save_processed_data:

            processed_data_path = os.environ["PROCESSED_DATA_PATH"]

            X_train.to_csv(processed_data_path + "X_train.csv", index=False)
            X_test.to_csv(processed_data_path + "X_test.csv", index=False)
            y_train.to_csv(processed_data_path + "y_train.csv", index=False)
            y_test.to_csv(processed_data_path + "y_test.csv", index=False)

            logging.info(f"Saved processed data in {processed_data_path}")

        return X_train, X_test, y_train, y_test

    @staticmethod
    def full_pipeline(filename: str, save_processed_data=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        logging.info("Running the full pipeline for the model")

        df = FeatureBuilder.read_data(filename)
        X, y = FeatureBuilder.build_features(df)
        X_train, X_test, y_train, y_test = FeatureBuilder.get_train_test_sets(X, y, save_processed_data)

        logging.info("Finished running the pipeline")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    filename = os.environ["RAW_DATA_FILENAME"]
    X_train, X_test, y_train, y_test = FeatureBuilder.full_pipeline(filename=filename, save_processed_data=True)
