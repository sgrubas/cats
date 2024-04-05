from pydantic import BaseModel
from typing import Union, Any, Callable, Dict
from cats.metrics import linalg_metric_func, binary_metric_func, find_word_starting_with
import numpy as np
import inspect
from functools import wraps


# TODO:
#   - picks introspection for Detector is needed


class BaseScoring(BaseModel):
    operator: Any
    generator_x_and_y: Union[Callable, Any]
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

    def evaluate(self, x_and_y=None):
        x_and_y = self.parse_data_generator(self.generator_x_and_y if x_and_y is None else x_and_y)

        metrics = {name: [] for name in self.metric_functions.keys()}
        metrics["Elapsed_time_sec"] = []

        for xi, yi, *attrs in x_and_y:
            attrs = attrs or [{}]
            if self.prepare_operator is not None:
                self.prepare_operator(self.operator, xi)
            operator_result = self.operator * xi
            for name, func in self.metric_functions.items():
                kwargs = {kw: attrs[0].get(kw, None) for kw in inspect.getfullargspec(func).args[2:]}

                metrics[name].append(func(yi, operator_result, **kwargs))
            metrics["Elapsed_time_sec"].append(operator_result.history.history['Total'])

        return metrics


class DenoiserScoring(BaseScoring):
    metric_functions: Union[str, Callable, Dict[str, Callable], Dict[str, str]] = "rel l2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_metric_functions()

    @staticmethod
    def metric_wrapper(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(yi, operator_result, *args, **kwargs):
            return func(yi, operator_result.signal_denoised, *args[2:], **kwargs)
        wrapper.__signature__ = sig

        return wrapper

    def set_metric_functions(self):
        if isinstance(self.metric_functions, str):
            func = self.metric_wrapper(linalg_metric_func(self.metric_functions, axis=-1))
            self.metric_functions = {self.metric_functions: func}

        elif callable(self.metric_functions):
            self.metric_functions = {"metric": self.metric_wrapper(self.metric_functions)}

        elif isinstance(self.metric_functions, dict):
            metric_functions = {}

            for name, func_or_str in self.metric_functions.items():
                if callable(func_or_str):
                    func = func_or_str
                elif isinstance(func_or_str, str):
                    func = linalg_metric_func(func_or_str, axis=-1)
                else:
                    raise ValueError(f"Passed metric function {name} is of unknown type {type(func_or_str) = }")
                metric_functions[name] = self.metric_wrapper(func)
            self.metric_functions = metric_functions

        else:
            raise ValueError(f"Unknown input type for {type(self.metric_functions) = }")


class DetectorScoring(BaseScoring):
    metric_functions: Union[str, Callable, Dict[str, Callable], Dict[str, str]] = "picks2.5 f0.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_metric_functions()

    @staticmethod
    def metric_wrapper(func, picks=True):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(yi, operator_result, *args, **kwargs):
            if picks:
                features = operator_result.picked_features
                y_pred = np.empty(features.shape, dtype=object)
                for ind in np.ndindex(*features.shape):
                    y_pred[ind] = features[ind][:, 0]
            else:
                raise NotImplementedError("Metrics not based on pick haven't been implemented yet")
                # y_pred = ProjectIntervals(detector_result.detected_intervals)

            return func(yi, y_pred, *args[2:], **kwargs)

        wrapper.__signature__ = sig

        return wrapper

    def set_metric_functions(self):
        if isinstance(self.metric_functions, str):
            picks = find_word_starting_with(self.metric_functions, "pick", case_insensitive=True)
            func = self.metric_wrapper(binary_metric_func(self.metric_functions), picks)
            self.metric_functions = {self.metric_functions: func}

        elif callable(self.metric_functions):
            self.metric_functions = self.metric_wrapper(self.metric_functions,
                                                        picks=True)  # NOTE: introspection for 'picks' !!

        elif isinstance(self.metric_functions, dict):
            metric_functions = {}
            for name, func_or_str in self.metric_functions.items():
                if callable(func_or_str):
                    func = func_or_str
                    picks = True  # NOTE: introspection for 'picks' !!

                elif isinstance(func_or_str, str):
                    func = binary_metric_func(func_or_str)
                    picks = find_word_starting_with(func_or_str, "pick", case_insensitive=True)

                else:
                    raise ValueError(f"Passed metric function {name} is of unknown type {type(func_or_str) = }")

                metric_functions[name] = self.metric_wrapper(func, picks=picks)

            self.metric_functions = metric_functions

        else:
            raise ValueError(f"Unknown input type for {type(self.metric_functions) = }")
