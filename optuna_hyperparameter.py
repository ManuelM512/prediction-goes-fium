import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from model_hyperp_suggestion import (
    get_rf_with_parameters,
    get_mnb_with_parameters,
    get_gb_with_parameters,
)


def choose_model(trial, opt: int):
    """
    Choose and return a machine learning model based on the given option.
    All of them already had the hyperparemeters suggested.

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
            return get_mnb_with_parameters(trial)
        case 2:
            return get_gb_with_parameters(trial)
        case 3:
            return get_rf_with_parameters(trial)
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def hyperparemeterization(X, y, model_opt):
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = choose_model(trial, model_opt)

        # Evaluate the model using cross-validation
        score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)
        accuracy = score.mean()

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    return study.best_trial, study.best_params
