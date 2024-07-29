from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from dataset_prep import pre_proc


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


# TODO: Mejorar uso del nombre del dataset, y creación si no existe
def train(dataset_name):
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
    df = pre_proc(dataset_name)
    X = df[features]
    y = df.positionOrder.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model
