import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import mlflow
import random
import numpy as np

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow server --host 127.0.0.1 --port 8080


def load_datasets():
    folder_path = "./datasets"

    # ID of constructors
    constructors_df = pd.read_csv(folder_path + "/constructors.csv")

    # ID of drivers
    drivers_df = pd.read_csv(folder_path + "/drivers.csv")

    # ID of each race
    races = pd.read_csv(folder_path + "/races.csv")

    # Starting grid and final position of every driver in every race (also includes status of each race)
    results_df = pd.read_csv(folder_path + "/results.csv")

    return constructors_df, drivers_df, races, results_df


def pre_proc():
    constructors_df, drivers_df, races, results_df = load_datasets()
    # Extract only relevant information about the race for training purposes
    race_df = races[["raceId", "year", "round", "circuitId"]].copy()
    # "(...) before 1981, the cars in F1 are drastically different from today."
    race_df = race_df[race_df["year"] >= 1982]

    res_df = results_df[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder"]
    ].copy()
    df = pd.merge(race_df, res_df, on="raceId")
    df_final = df.drop(labels=["raceId"], axis=1)
    return df_final


def train():
    features = ["year", "round", "circuitId", "driverId", "constructorId", "grid"]
    df = pre_proc()
    df = df.dropna()
    X = df[features]
    y = df.positionOrder.astype(int)
    test_size = 0.2
    random_state = 42
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        n_estimators = 100
        mlflow.log_param("features", features)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        # Initialize the Random Forest Regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_regressor.predict(X_test).astype(int)

        # Log metrics
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)

        # Number of observations and predictors
        n = len(y_test)
        k = X.shape[1]  # Number of predictors

        # Calculate Adjusted R-squared
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        # Log metrics using mlflow
        mlflow.log_metric("Mean Absolute Error - MAE", mae)
        mlflow.log_metric("Mean Squared Error - MSE", mse)
        mlflow.log_metric("Root Mean Squared Error - RMSE", rmse)
        mlflow.log_metric("R-squared - R2", r2)
        mlflow.log_metric("Adjusted R-squared - R2_adj", adjusted_r2)
        mlflow.log_metric("Explained Variance Score", explained_variance)



        # Log the model
        model_info = mlflow.sklearn.log_model(rf_regressor, "model_rf")

    return model_info


def predictor():
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(train().model_uri)

    # Values for the specified columns
    year = 2024
    round_number = 14
    circuit_id = 13
    driver_ids = [
        1,
        50,
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
        76,
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
    # Create a list of integers from 1 to 20
    numbers = [3,11,6, 20, 2, 13, 5, 18, 19, 4, 10, 14, 16, 17, 9, 12, 1, 7, 15, 8]

    # Shuffle the list to get a random order
    random.shuffle(numbers)
    # Create the dataset
    data = {
        "year": [year] * len(driver_ids),
        "round": [round_number] * len(driver_ids),
        "circuitId": [circuit_id] * len(driver_ids),
        "driverId": driver_ids,
        "constructorId": constructor_ids,
        "grid": numbers,
    }

    df = pd.DataFrame(data)

    pred = loaded_model.predict(df)

    print(pred)


predictor()
