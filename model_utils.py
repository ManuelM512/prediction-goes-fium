from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from dataset_prep import pre_proc, check_dataset_exists, load_dataframe
from optuna_hyperparameter import hyperparemeterization
from sklearn.model_selection import train_test_split
from experiment_tracking_utils import track_experiment


def create_model(opt: int, params: dict):
    """
    Choose and return a machine learning model based on the given option.

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
            return MultinomialNB(alpha=params["alpha"], fit_prior=params["fit_prior"])
        case 2:
            return GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42,
            )
        case 3:
            return RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42,
            )
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def predict_unique(model, to_predict_df):
    probabilities = model.predict_proba(to_predict_df)

    unique_position_orders = list(range(1, 21))
    predictions = []
    # TODO: Make n iterations of this, in order to get an avg position for each driver
    # Iterate through each sample's predicted probabilities
    for i, prob in enumerate(probabilities):
        top_classes = np.argsort(prob)[::-1]
        # Find the first available unique positionOrder
        for cls in top_classes:
            if cls + 1 in unique_position_orders:
                predictions.append(cls + 1)
                unique_position_orders.remove(cls + 1)
                break

    return predictions


def get_model_name_by_opt(opt: int):
    match opt:
        case 1:
            return "MNB"
        case 2:
            return "GBC"
        case 3:
            return "RFC"
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def train(dataset_path: str, model_opt: int):
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
    trial, params = hyperparemeterization(X, y, model_opt)
    model = create_model(model_opt, params)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)
    model_name = get_model_name_by_opt(model_opt)
    # Predict on the test data
    y_pred = model.predict(X_test)
    track_experiment(model, model_name, y_pred, list(y_test), params)

    return model
