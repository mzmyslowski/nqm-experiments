import numpy as np
import sklearn


def get_flattened_target_variance(y_true, y_pred):
    return y_true.var()


def get_mape(y_true, y_pred):
    y_diff = np.subtract(y_true, y_pred)
    y_diff_normalised = np.divide(y_diff, y_true)
    y_diff_normalised_abs = np.abs(y_diff_normalised)
    mape = np.mean(y_diff_normalised_abs)
    return mape


def get_mae(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'Shapes must agree. Got 1. {} and 2. {}'.format(y_true.shape, y_pred.shape)
    return sklearn.metrics.mean_absolute_error(y_true=y_true.reshape(-1), y_pred=y_pred.reshape(-1))


def get_normalised_rmse(y_true, y_pred):
    rmse = get_rmse(y_true=y_true, y_pred=y_pred)
    return rmse / np.std(y_true)


def get_rmse(y_true, y_pred):
    mse = get_mse(y_true=y_true, y_pred=y_pred)
    return np.sqrt(mse)


def get_mse(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'Shapes must agree. Got 1. {} and 2. {}'.format(y_true.shape, y_pred.shape)
    return sklearn.metrics.mean_squared_error(y_true=y_true.reshape(-1), y_pred=y_pred.reshape(-1))


class MetricsFactory:
    METRICS_FUNCS = {
        'Target variance': get_flattened_target_variance,
        'MAE': get_mae,
        'MSE': get_mse
    }

    def __init__(self, metrics_names):
        self.metrics_names = metrics_names

    def evaluate_metrics(self, y, y_pred, prefix_name: str = ''):
        evaluated_metrics = {}
        for metric_name in self.metrics_names:
            metric_eval = self.METRICS_FUNCS[metric_name](y_true=y, y_pred=y_pred)
            evaluated_metrics['{}_{}'.format(prefix_name, metric_name)] = metric_eval.item()
        return evaluated_metrics
