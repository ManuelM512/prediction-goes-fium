import pandas as pd
def load_dataset(path):
    circuits = pd.read_csv(f'{path}/circuits.csv')
    status = pd.read_csv(f'{path}/status.csv')
    lap_times = pd.read_csv(f'{path}/lap_times.csv')
    sprint_results = pd.read_csv(f'{path}/sprint_results.csv')
    drivers = pd.read_csv(f'{path}/drivers.csv')
    races = pd.read_csv(f'{path}/races.csv')
    constructors = pd.read_csv(f'{path}/constructors.csv')
    constructor_standings = pd.read_csv(f'{path}/constructor_standings.csv')
    qualifying = pd.read_csv(f'{path}/qualifying.csv')
    driver_standings = pd.read_csv(f'{path}/driver_standings.csv')
    constructor_results = pd.read_csv(f'{path}/constructor_results.csv')
    pit_stops = pd.read_csv(f'{path}/pit_stops.csv')
    seasons = pd.read_csv(f'{path}/seasons.csv')
    results = pd.read_csv(f'{path}/results.csv')
    