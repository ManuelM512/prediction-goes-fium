import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import mlflow
from dataset_prep import pre_proc

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow server --host 127.0.0.1 --port 8080

# TODO: Refactor, move functions and shorten them
# TODO: Function to get top 3, driver performance ... from the dataset


def model_train_test(df, features):
    X = df[features]
    y = df.positionOrder.astype(int)
    return X, y


def train(dataset_name):
    features = [
        "year",
        "round",
        "circuitId",
        "driverId",
        "constructorId",
        "grid",
        "Top 3 Finish",
        "Driver Top 3 Finish Percentage (Last Year)",
        "Constructor Top 3 Finish Percentage (Last Year)",
    ]
    df = pre_proc(dataset_name)
    X, y = model_train_test(df, features)

    # Start an MLflow run
    with mlflow.start_run():
        mnb = MultinomialNB()
        mnb.fit(X, y)
        # Log the model

        model_info = mlflow.sklearn.log_model(mnb, "mnb_first")

    return model_info


def pretty_printer(pred_data, driver_ids):
    pred_df = pd.DataFrame(pred_data, columns=["result"])
    driver_ids_df = pd.DataFrame(driver_ids, columns=["driverId"])
    driver_info = pd.read_csv("./datasets/drivers.csv")
    merged_df = pd.merge(driver_ids_df, driver_info, on="driverId", how="inner")
    merged_df["driverName"] = merged_df["forename"] + " " + merged_df["surname"]
    final_df = pd.concat([merged_df.reset_index(drop=True), pred_df], axis=1)
    final_df = final_df[["driverId", "driverName", "result"]]

    print(final_df)
    return final_df


def get_data():
    year = 2024
    round_number = 14
    circuit_id = 13
    driver_ids = [
        1,
        830,
        847,
        852,
        815,
        817,
        857,
        858,
        855,
        846,
        848,
        822,
        807,
        825,
        839,
        842,
        844,
        832,
        840,
        4,
    ]
    constructor_ids = [
        131,
        9,
        131,
        215,
        9,
        215,
        1,
        3,
        15,
        1,
        3,
        15,
        210,
        210,
        214,
        214,
        6,
        6,
        117,
        117,
    ]
    numbers = [3, 11, 6, 20, 2, 13, 5, 18, 19, 4, 10, 14, 16, 17, 9, 12, 1, 7, 15, 8]

    # Create the dataset to predict
    data = {
        "year": [year] * len(driver_ids),
        "round": [round_number] * len(driver_ids),
        "circuitId": [circuit_id] * len(driver_ids),
        "driverId": driver_ids,
        "constructorId": constructor_ids,
        "grid": numbers,
    }
    #    "Top 3 Finish": ,
    #    "Driver Top 3 Finish Percentage (Last Year)": ,
    #    "Constructor Top 3 Finish Percentage (Last Year)": ,

    return data


def get_complementary_data(data: dict):
    driver_ids = data["driverId"]
    year = data["year"][0]
    constructor_ids = data["constructorId"]
    # ch_round = data["round"][0]

    results_df = pd.read_csv("./datasets/results.csv")
    races_df = pd.read_csv("./datasets/races.csv")
    results_x_year_df = pd.merge(results_df, races_df, on="raceId", how="inner")[
        ["year", "driverId", "round", "raceId", "positionOrder", "constructorId"]
    ]
    top_3_finishes = []
    for driver_id in driver_ids:
        driver_races = results_x_year_df[results_x_year_df["driverId"] == driver_id]
        top3s = driver_races["positionOrder"].le(3).astype(int).sum()
        top_3_finishes.append(top3s)

    prior_year_df = results_x_year_df[results_x_year_df["year"] == year-1]

    top_3_per_last_year = []
    for driver_id in driver_ids:
        driver_races = prior_year_df[prior_year_df["driverId"] == driver_id]
        race_count = driver_races.shape[0]
        top3s = driver_races["positionOrder"].le(3).astype(int).sum()
        top_3_per_last_year.append((top3s/race_count)*100)

    # TODO: Optimize this, its counting 2 times the same constructor id...
    top_3_constructors_per_last_year = []
    for constructor_id in constructor_ids:
        constructor_races= prior_year_df[prior_year_df["constructorId"] == constructor_id]
        race_count = constructor_races.shape[0]
        top3s = constructor_races["positionOrder"].le(3).astype(int).sum()
        top3s_per_constructor = 0
        if race_count > 0:
            top3s_per_constructor = (top3s / race_count) * 100
        top_3_constructors_per_last_year.append(top3s_per_constructor)

    data["Top 3 Finish"] = top_3_finishes
    data["Driver Top 3 Finish Percentage (Last Year)"] = top_3_per_last_year
    data["Constructor Top 3 Finish Percentage (Last Year)"] = top_3_constructors_per_last_year

    return data


def predictor():
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(train("mnb_first").model_uri)
    data = get_complementary_data(get_data())
    driver_ids = data["driverId"]
    df = pd.DataFrame(data)
    pred = loaded_model.predict(df)
    final_df = pretty_printer(pred, driver_ids)

    return final_df


predictor()
