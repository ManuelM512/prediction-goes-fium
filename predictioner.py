import pandas as pd
from dataset_prep import get_df_to_print
from prediction_data_loader import (
    get_complementary_data,
    get_data,
)
from model_utils import predict_unique, train
from experiment_tracking_utils import track_experiment


def positioner():
    dataset_folder = "./model_datasets/"
    loaded_model = train(f"{dataset_folder}test_model", 1)
    race_data = get_data("2024_spa")
    # The dataset could have orderPosition in it, which is the target and not for train
    race_data_to_predict = race_data[
        ["year", "round", "circuitId", "driverId", "constructorId", "grid"]
    ]
    data = get_complementary_data(race_data_to_predict)
    driver_ids = data["driverId"]
    to_predict_df = pd.DataFrame(data)
    pred_results = predict_unique(loaded_model, to_predict_df)
    # If the race was already done, take it as an experiment for mlflow
    if "positionOrder" in race_data.columns:
        actual_results = list(race_data["positionOrder"])
        track_experiment(loaded_model, "test_model", pred_results, actual_results)
    # In any case, it prints the predictions
    driver_result_df = get_df_to_print(pred_results, driver_ids)

    return driver_result_df


if __name__ == "__main__":
    print(positioner())
