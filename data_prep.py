import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_datasets():
    folder_path = './datasets'

    # ID of constructors
    constructors_df = pd.read_csv(folder_path + '/constructors.csv')

    # ID of drivers
    drivers_df = pd.read_csv(folder_path + '/drivers.csv')

    # ID of each race
    races = pd.read_csv(folder_path + '/races.csv')

    # Starting grid and final position of every driver in every race (also includes status of each race)
    results_df = pd.read_csv(folder_path + '/results.csv')

    return constructors_df, drivers_df, races, results_df


def pre_proc():
    constructors_df, drivers_df, races, results_df = load_datasets()
    # Extract only relevant information about the race for training purposes
    race_df = races[["raceId", "year", "round", "circuitId"]].copy()
    # "(...)before 1981, the cars in F1 are drastically different from today."
    race_df = race_df[race_df["year"] >= 1982]

    res_df = results_df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']].copy()
    df = pd.merge(race_df, res_df, on='raceId')
    df_final = df.drop(labels=["raceId"], axis=1)
    return df_final

def train():
    features = ['year', 'round', 'circuitId', 'driverId', 'constructorId', 'grid']
    df = pre_proc()
    df = df.dropna()
    X = df[features]
    y = df.positionOrder
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_regressor.fit(X_train, y_train)
    #print(X_test)
    # Make predictions
    #y_pred = rf_regressor.predict(X_test)
    new_data = {
        'year': 2024,
        'round': 14,
        'circuitId': 13,
        'driverId': 844,
        'constructorId': 6,
        'grid': 10
    }
    new_data_df = pd.DataFrame([new_data])
    y_pred = rf_regressor.predict(new_data_df)
    return y_pred

print(train())
