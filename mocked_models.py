from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from dataset_prep import pre_proc, check_dataset_exists, load_dataframe
from optuna_hyperparameter import hyperparemeterization
from sklearn.model_selection import train_test_split
from experiment_tracking_utils import track_experiment


def create_mocked_model(opt: int):
    """
    Choose and return a machine learning model, based on the given option, with the best parameters taken by hyperparameterization.

    Parameters:
    -----------
    opt : int
        An integer representing the model choice:
        - 1: Multinomial Naive Bayes
        - 2: Gradient Boosting Classifier
        - 3: Random Forest Classifier
    """
    match opt:
        case 1:
            return MultinomialNB(alpha=0.0012523174615980547, fit_prior=False)
        case 2:
            return GradientBoostingClassifier()
        case 3:
            return RandomForestClassifier(
                n_estimators=175,
                max_depth=5,
                min_samples_split=9,
                min_samples_leaf=10,
                random_state=42,
            )
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def mocked_train(model_opt: int, dataset_path: str):
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
    if not check_dataset_exists(dataset_path):
        df = pre_proc(dataset_path)
    else:
        df = load_dataframe(dataset_path)
    X = df[features]
    y = df.positionOrder.astype(int)
    model = create_mocked_model(model_opt)
    model.fit(X, y)

    return model
