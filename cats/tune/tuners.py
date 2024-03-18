from pydantic import BaseModel
from typing import Union, Any, Callable, Dict
from cats.metrics import linalg_metric_func, binary_metric_func, find_word_starting_with
import numpy as np
import inspect


# TODO:
#  - memory usage is another metric (memory profiling?)


class BaseScoring(BaseModel):
    operator: Any
    x_generator: Union[Callable, Any]
    y_generator: Union[Callable, Any]
    metric_functions: Dict[str, Callable]
    fixed_params: dict = {}
    param_parser: Callable = None
    prepare_operator: Callable = None

    def export_main_params(self):
        return {kw: getattr(self, kw, None) for kw in type(self).__fields__.keys()}

    def update_operator(self, **kwargs):
        if self.param_parser is not None:
            kwargs = self.param_parser(kwargs)
        kwargs.update(self.fixed_params)
        self.operator.reset_params(**kwargs)

    @staticmethod
    def parse_data_generator(data_generator):
        if inspect.isgeneratorfunction(data_generator) or callable(data_generator):
            return data_generator()
        else:
            return data_generator

    def evaluate(self, x=None, y=None):
        x = self.parse_data_generator(self.x_generator if x is None else x)
        y = self.parse_data_generator(self.y_generator if y is None else y)

        metrics = {name: [] for name in self.metric_functions.keys()}
        metrics["Elapsed_time_sec"] = []

        for xi, yi in zip(x, y):
            if self.prepare_operator is not None:
                self.prepare_operator(self.operator, xi)
            operator_result = self.operator * xi
            for name, func in self.metric_functions.items():
                metrics[name].append(func(yi, operator_result))
                metrics["Elapsed_time_sec"].append(operator_result.history.history['Total'])

        return metrics


class DenoiserScoring(BaseScoring):
    metric_functions: Union[str, Callable, Dict[str, Callable]] = "rel l2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_metric_function()

    def set_metric_function(self):
        if isinstance(self.metric_functions, str):
            func = linalg_metric_func(self.metric_functions, axis=-1)
            negative = bool(find_word_starting_with(self.metric_functions, "neg", case_insensitive=True))
            sign = -1.0 if negative else 1.0

            def _metric_func(y_true, denoising_result):
                return sign * func(y_true, denoising_result.signal_denoised)

            self.metric_functions = {self.metric_functions: _metric_func}

        elif callable(self.metric_functions):
            self.metric_functions = {"metric": self.metric_functions}

        elif isinstance(self.metric_functions, dict):
            pass

        else:
            raise ValueError(f"Unknown input type for {type(self.metric_functions) = }")


class DetectorScoring(DenoiserScoring):
    metric_functions: Union[str, Callable, Dict[str, Callable]] = "picks2.5 f0.5"

    def set_metric_function(self):
        if isinstance(self.metric_functions, str):
            picks = find_word_starting_with(self.metric_functions, "pick", case_insensitive=True)
            func = binary_metric_func(self.metric_functions)
            negative = bool(find_word_starting_with(self.metric_functions, "neg", case_insensitive=True))
            sign = -1.0 if negative else 1.0

            def _metric_func(y_true, detector_result):
                if picks:
                    features = detector_result.picked_features
                    y_pred = np.empty(features.shape, dtype=object)
                    for ind in np.ndindex(*features.shape):
                        y_pred[ind] = features[ind][:, 0]
                else:
                    raise NotImplementedError(f"Metric {self.metric_functions} has not been implemented yet")
                    # y_pred = ProjectIntervals(detector_result.detected_intervals)

                return sign * func(y_true, y_pred)

            self.metric_functions = {self.metric_functions: _metric_func}

        elif callable(self.metric_functions):
            self.metric_functions = {"metric": self.metric_functions}

        elif isinstance(self.metric_functions, dict):
            pass

        else:
            raise ValueError(f"Unknown input type for {type(self.metric_functions) = }")

