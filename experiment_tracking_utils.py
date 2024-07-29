from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow server --host 127.0.0.1 --port 8080


def difference_in_pred(actual, pred):
    dif_list = []
    for i in range(len(actual)):
        dif_list.append(abs(actual[i] - pred[i]))
    dif_list.sort()
    return dif_list


def list_median(l: list):
    len_l = len(l)
    med = len_l // 2
    if len_l % 2 == 0:
        return (l[med] + l[med - 1]) / 2
    return l[med]


def get_metrics(actual, pred):
    dif_list = difference_in_pred(actual, pred)
    metric_dict = {
        "accuracy_score": accuracy_score(actual, pred),
        "precision_score": precision_score(actual, pred, average="weighted"),
        "recall_score": recall_score(actual, pred, average="weighted"),
        "f1_score": f1_score(actual, pred, average="weighted"),
        "median_error": list_median(dif_list),
        "avg_error": sum(dif_list) / len(dif_list),
        "error_sum": sum(dif_list),
    }
    return metric_dict


def log_to_mlflow(model, model_name: str, metrics_dict: dict, params_dict: dict):
    with mlflow.start_run():
        for metric in metrics_dict:
            mlflow.log_metric(f"{metric}", metrics_dict[metric])
        for param in params_dict:
            mlflow.log_param(f"{param}", params_dict[param])
        model_info = mlflow.sklearn.log_model(model, model_name)
    return model_info


def track_experiment(model, model_name, pred, actual):
    metrics_dict = get_metrics(actual, pred)
    params_dict = {"test_size": 0.2, "random_state": 42}
    log_to_mlflow(model, model_name, metrics_dict, params_dict)
