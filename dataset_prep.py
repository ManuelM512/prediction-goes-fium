import pandas as pd
from os import path


def load_datasets():
    folder_path = "./datasets"
    constructors_df = pd.read_csv(folder_path + "/constructors.csv")
    drivers_df = pd.read_csv(folder_path + "/drivers.csv")
    races = pd.read_csv(folder_path + "/races.csv")
    results_df = pd.read_csv(folder_path + "/results.csv")

    return constructors_df, drivers_df, races, results_df


def feature_engineering(df):
    # These are taken from
    # https://www.kaggle.com/code/yanrogerweng/formula-1-race-prediction#%22Top-3-Finish%22:-Add-Target-Variable
    # Creating a column for Top 3 Finish
    df["Top 3 Finish"] = df["positionOrder"].le(3).astype(int)

    # Calculating the total number of races and top 3 finishes for each driver in each year
    driver_yearly_stats = (
        df.groupby(["year", "driverId"])
        .agg(Total_Races=("raceId", "nunique"), Top_3_Finishes=("Top 3 Finish", "sum"))
        .reset_index()
    )

    # Calculating the percentage of top 3 finishes for each driver in each year
    driver_yearly_stats["Driver Top 3 Finish Percentage (This Year)"] = (
        driver_yearly_stats["Top_3_Finishes"] / driver_yearly_stats["Total_Races"]
    ) * 100
    # Shifting the driver percentages to the next year for last year's data
    driver_last_year_stats = driver_yearly_stats.copy()
    driver_last_year_stats["year"] += 1
    driver_last_year_stats = driver_last_year_stats.rename(
        columns={
            "Driver Top 3 Finish Percentage (This Year)": "Driver Top 3 Finish Percentage (Last Year)"
        }
    )

    df = pd.merge(
        df,
        driver_last_year_stats[
            ["year", "driverId", "Driver Top 3 Finish Percentage (Last Year)"]
        ],
        on=["year", "driverId"],
        how="left",
    )

    # Calculating mean of top 3 finishes percentages for the two drivers in each constructor last year
    constructor_last_year_stats = (
        df.groupby(["year", "constructorId", "round"])
        .agg(
            Sum_Top_3_Finishes_Last_Year=(
                "Driver Top 3 Finish Percentage (Last Year)",
                "sum",
            )
        )
        .reset_index()
    )

    # Calculating the percentage of top 3 finishes for each constructor last year
    constructor_last_year_stats["Constructor Top 3 Finish Percentage (Last Year)"] = (
        constructor_last_year_stats["Sum_Top_3_Finishes_Last_Year"] / 2
    )

    df = pd.merge(
        df,
        constructor_last_year_stats[
            [
                "year",
                "constructorId",
                "round",
                "Constructor Top 3 Finish Percentage (Last Year)",
            ]
        ],
        on=["year", "constructorId", "round"],
        how="left",
    )

    # From here and after, was me trying new things
    # Nothing so far :(

    return df


def pre_proc(dataset_path):
    constructors_df, drivers_df, races, results_df = load_datasets()
    # Extract only relevant information about the race for training purposes
    race_df = races[["raceId", "year", "round", "circuitId"]].copy()
    # "(...) before 1981, the cars in F1 are drastically different from today."
    race_df = race_df[race_df["year"] >= 1982]
    results_df = results_df[
        (results_df["positionOrder"] < 21) & (results_df["positionOrder"] > 0)
    ]
    res_df = results_df[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder"]
    ].copy()
    df = pd.merge(race_df, res_df, on="raceId")
    df = feature_engineering(df)
    df_final = df.drop(labels=["raceId"], axis=1)
    df_final = df_final.dropna()
    df_final = df_final
    df_final.to_csv(dataset_path)
    return df_final


def get_df_to_print(pred_data, driver_ids):
    pred_df = pd.DataFrame(pred_data, columns=["result"])
    driver_ids_df = pd.DataFrame(driver_ids, columns=["driverId"])
    driver_info = pd.read_csv("./datasets/drivers.csv")
    merged_df = pd.merge(driver_ids_df, driver_info, on="driverId", how="inner")
    merged_df["driverName"] = merged_df["forename"] + " " + merged_df["surname"]
    final_df = pd.concat([merged_df.reset_index(drop=True), pred_df], axis=1)
    final_df = final_df[["driverName", "result"]].sort_values(by="result")

    return final_df


def check_dataset_exists(dataset_path: str):
    return path.exists(dataset_path)


def load_dataframe(dataset_path: str):
    df = pd.read_csv(dataset_path)
    return df
