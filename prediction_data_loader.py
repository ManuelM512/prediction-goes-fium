import pandas as pd


def get_data(race_name):
    df_to_predict = pd.read_csv(f"./race_results/{race_name}")
    return df_to_predict


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

    prior_year_df = results_x_year_df[results_x_year_df["year"] == year - 1]

    top_3_per_last_year = []
    for driver_id in driver_ids:
        driver_races = prior_year_df[prior_year_df["driverId"] == driver_id]
        race_count = driver_races.shape[0]
        top3s = driver_races["positionOrder"].le(3).astype(int).sum()
        top_3_per_last_year.append((top3s / race_count) * 100)

    top_3_constructors_per_last_year = []
    constructor_top_3_per_dict = {}
    for constructor_id in constructor_ids:
        top3s_per_constructor = constructor_top_3_per_dict.get(constructor_id, -1)
        if top3s_per_constructor == -1:
            constructor_races = prior_year_df[
                prior_year_df["constructorId"] == constructor_id
            ]
            race_count = constructor_races.shape[0]
            top3s = constructor_races["positionOrder"].le(3).astype(int).sum()
            if race_count > 0:
                top3s_per_constructor = (top3s / race_count) * 100
            constructor_top_3_per_dict[constructor_id] = top3s_per_constructor
        top_3_constructors_per_last_year.append(top3s_per_constructor)

    data["Top 3 Finish"] = top_3_finishes
    data["Driver Top 3 Finish Percentage (Last Year)"] = top_3_per_last_year
    data["Constructor Top 3 Finish Percentage (Last Year)"] = (
        top_3_constructors_per_last_year
    )

    return data
