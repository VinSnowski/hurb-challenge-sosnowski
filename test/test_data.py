import unittest
import os, dotenv
from src.features.build_features import FeatureBuilder

dotenv.load_dotenv(".env")


class TestData(unittest.TestCase):
    def test_feature_building_pipeline(self):
        filename = os.environ["RAW_DATA_FILENAME"]
        df = FeatureBuilder.read_data(filename)
        X, y = FeatureBuilder.build_features(df)
        X_train, X_test, y_train, y_test = FeatureBuilder.get_train_test_sets(X, y)

        final_X_expected_columns = [
            "hotel",
            "meal",
            "market_segment",
            "distribution_channel",
            "reserved_room_type",
            "deposit_type",
            "customer_type",
            "year",
            "month",
            "day",
            "lead_time",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "agent",
            "company",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
        ]

        y_possible_values = set([0, 1])
        y_values = set(set(y))

        self.assertEqual(y_possible_values, y_values)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(df))
        self.assertEqual(len(y_train) + len(y_test), len(df))
        self.assertEqual(list(X_train.columns), final_X_expected_columns)
        self.assertEqual(list(X_test.columns), final_X_expected_columns)


if __name__ == "__main__":
    unittest.main()
