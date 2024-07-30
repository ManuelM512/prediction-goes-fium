import pandas as pd
from dataset_prep import get_df_to_print
from prediction_data_loader import (
    get_actual_position_order,
    get_complementary_data,
    get_data,
)
from model_utils import predict_unique, train
from experiment_tracking_utils import track_experiment


def positioner():
    dataset_folder = "./model_datasets/"
    loaded_model = train(f"{dataset_folder}test_model", 1)
    data = get_complementary_data(get_data())
    driver_ids = data["driverId"]
    to_predict_df = pd.DataFrame(data)
    pred_results = predict_unique(loaded_model, to_predict_df)
    actual_results = get_actual_position_order()
    track_experiment(loaded_model, "test_model", pred_results, actual_results)
    driver_result_df = get_df_to_print(pred_results, driver_ids)

    return driver_result_df


if __name__ == "__main__":
    print(positioner())
