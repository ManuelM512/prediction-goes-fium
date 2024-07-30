import pandas as pd


# TODO: La idea sería después cambiar a que esto se cargue directo desde jsons
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

    data_to_predict = {
        "year": [year] * len(driver_ids),
        "round": [round_number] * len(driver_ids),
        "circuitId": [circuit_id] * len(driver_ids),
        "driverId": driver_ids,
        "constructorId": constructor_ids,
        "grid": numbers,
    }

    return data_to_predict


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

    # TODO: Optimize this, its counting 2 times the same constructor id...
    top_3_constructors_per_last_year = []
    for constructor_id in constructor_ids:
        constructor_races = prior_year_df[
            prior_year_df["constructorId"] == constructor_id
        ]
        race_count = constructor_races.shape[0]
        top3s = constructor_races["positionOrder"].le(3).astype(int).sum()
        top3s_per_constructor = 0
        if race_count > 0:
            top3s_per_constructor = (top3s / race_count) * 100
        top_3_constructors_per_last_year.append(top3s_per_constructor)

    data["Top 3 Finish"] = top_3_finishes
    data["Driver Top 3 Finish Percentage (Last Year)"] = top_3_per_last_year
    data["Constructor Top 3 Finish Percentage (Last Year)"] = (
        top_3_constructors_per_last_year
    )

    return data


# TODO: No se por qué no lo puse directo con el get data jajaja
def get_actual_position_order():
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
    return actual_results
