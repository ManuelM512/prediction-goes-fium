import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import random

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


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


def feature_engineering(df):
    # This are taken from
    # https://www.kaggle.com/code/yanrogerweng/formula-1-race-prediction#%22Top-3-Finish%22:-Add-Target-Variable
    # Creating a column for Top 3 Finish
    df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)

    # Calculating the total number of races and top 3 finishes for each driver in each year
    driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
        Total_Races=('raceId', 'nunique'),
        Top_3_Finishes=('Top 3 Finish', 'sum')
    ).reset_index()

    # Calculating the percentage of top 3 finishes for each driver in each year
    driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] /
                                                                         driver_yearly_stats['Total_Races']) * 100
    # Shifting the driver percentages to the next year for last year's data
    driver_last_year_stats = driver_yearly_stats.copy()
    driver_last_year_stats['year'] += 1
    driver_last_year_stats = driver_last_year_stats.rename(
        columns={'Driver Top 3 Finish Percentage (This Year)': 'Driver Top 3 Finish Percentage (Last Year)'})

    df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Top 3 Finish Percentage (Last Year)']],
                  on=['year', 'driverId'], how='left')

    # Calculating mean of top 3 finishes percentages for the two drivers in each constructor last year
    constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
        Sum_Top_3_Finishes_Last_Year=('Driver Top 3 Finish Percentage (Last Year)', 'sum')
    ).reset_index()

    # Calculating the percentage of top 3 finishes for each constructor last year
    constructor_last_year_stats['Constructor Top 3 Finish Percentage (Last Year)'] = constructor_last_year_stats[
                                                                                         "Sum_Top_3_Finishes_Last_Year"] / 2

    df = pd.merge(df, constructor_last_year_stats[
        ['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']],
                  on=['year', 'constructorId', 'round'], how='left')

    # From here and after, was me trying new things
    # Nothing so far :(

    return df


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
    df = feature_engineering(df)
    df_final = df.drop(labels=["raceId"], axis=1)
    df_final = df_final.dropna()
    return df_final

# TODO: Refactor, move functions and shorten them
# TODO: Function to get top 3, driver performance ... from the dataset

def train():
    features = ["year", "round", "circuitId", "driverId", "constructorId", "grid", "Top 3 Finish", "Driver Top 3 Finish Percentage (Last Year)", "Constructor Top 3 Finish Percentage (Last Year)", ]
    df = pre_proc()
    dataset_name = "first_rf"
    df.to_csv(f"./model_datasets/{dataset_name}")
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
        mlflow.log_param("dataset_file", dataset_name)
        # Initialize the Random Forest Regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_regressor.predict(X_test).astype(int)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

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
    numbers = list(range(1, 21))

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
