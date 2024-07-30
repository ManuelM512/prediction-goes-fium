from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from dataset_prep import pre_proc, check_dataset_exists, load_dataframe


def choose_model(opt: int):
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
            return MultinomialNB()
        case 2:
            return GradientBoostingClassifier()
        case 3:
            return RandomForestClassifier()
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def predict_unique(model, to_predict_df):
    probabilities = model.predict_proba(to_predict_df)

    unique_position_orders = list(range(1, 21))
    predictions = []

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = choose_model(model_opt)
    model.fit(X_train, y_train)

    return model
