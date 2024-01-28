import numpy as np
import re


def calculate_metric(y_true, y_pred, metric_name, axis=-1):
    return metric_func(metric_name, axis=axis)(y_true, y_pred)


def metric_func(metric_name, axis=-1):
    relative = bool(find_word_starting_with(metric_name, "rel", case_insensitive=True))
    accuracy = bool(find_word_starting_with(metric_name, "acc", case_insensitive=True))

    order = find_word_starting_with(metric_name, "inf", case_insensitive=True)
    if len(order) < 1:
        order = re.findall(r"-?\d+\.?\d*", metric_name)
    if len(order) != 1:
        raise ValueError(f"One type of metric must be specified, but given {order}")

    order = int(order[0]) if order[0].isnumeric() else float(order[0])

    err_f = accuracy_exp_linalg_norm if accuracy else error_linalg_norm

    def _err_func(y_true, y_pred):
        return err_f(y_true, y_pred, relative=relative, metric_ord=order, axis=axis)

    return _err_func


def error_linalg_norm(y_true, y_pred, relative=True, metric_ord=2, axis=-1):
    err = np.linalg.norm(y_true - y_pred, ord=metric_ord, axis=axis)
    if relative:
        err /= np.linalg.norm(y_true, ord=metric_ord, axis=axis)
    return err.mean()


def accuracy_exp_linalg_norm(y_true, y_pred, relative=True, metric_ord=2, axis=-1):
    err = error_linalg_norm(y_true, y_pred, relative=relative, metric_ord=metric_ord, axis=axis)
    acc = np.exp(-err)
    return acc


def find_word_starting_with(text, startwith, case_insensitive=True):
    if case_insensitive:
        letters = map(lambda x: x.lower() + x.upper(), startwith)
    else:
        letters = startwith
    letters = r''.join(map(lambda x: r"[{0}]".format(x), letters))
    pattern = r"-?\b{0}\w*\b".format(letters)

    return re.findall(pattern, text)
