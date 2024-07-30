import optuna
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def get_rf_with_parameters(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    return rf


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
            return get_rf_with_parameters()
        case _:
            raise ValueError(f"Invalid option {opt}. Expected values are 1, 2, or 3.")


def hyperparemeterization(X, y, model_opt):
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = choose_model(model_opt)

        # Evaluate the model using cross-validation
        score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)
        accuracy = score.mean()

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    return study.best_trial, study.best_params
