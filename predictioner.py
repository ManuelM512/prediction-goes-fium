import pandas as pd
from dataset_prep import get_df_to_print
from prediction_data_loader import get_complementary_data, get_data
from model_utils import predict_unique, train
from experiment_tracking_utils import track_experiment
# TODO: Traer la data real de otra manera


def positioner():
    loaded_model = train("mnb_first")
    data = get_complementary_data(get_data())
    driver_ids = data["driverId"]
    to_predict_df = pd.DataFrame(data)
    pred_results = predict_unique(loaded_model, to_predict_df)
    actual_results = [
        1,
        4,
        20,
        16,
        7,
        10,
        2,
        17,
        19,
        5,
        12,
        15,
        18,
        14,
        9,
        13,
        3,
        6,
        11,
        8,
    ]
    track_experiment(loaded_model, "prueba_mnb", pred_results, actual_results)
    driver_result_df = get_df_to_print(pred_results, driver_ids)

    return driver_result_df


if __name__ == "__main__":
    print(positioner())
