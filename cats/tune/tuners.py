from pydantic import BaseModel, Extra
from typing import Union, Any, Callable
from cats.metrics import linalg_metric_func, binary_metric_func, find_word_starting_with
import numpy as np
import inspect
# from bayes_opt import BayesianOptimization


class BaseScoring(BaseModel, extra=Extra.allow):
    operator: Any
    x_generator: Union[Callable, Any]
    y_generator: Union[Callable, Any]
    metric_function: Callable
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

    def evaluate_metric(self, x=None, y=None):
        x = self.parse_data_generator(self.x_generator if x is None else x)
        y = self.parse_data_generator(self.y_generator if y is None else y)

        metrics = []
        elapsed_time = []
        for xi, yi in zip(x, y):
            if self.prepare_operator is not None:
                self.prepare_operator(self.operator, xi)
            operator_result = self.operator * xi
            metrics.append(self.metric_function(yi, operator_result))
            elapsed_time.append(operator_result.history.history['Total'])

        return {'Metric': metrics, "Elapsed_time_sec": elapsed_time}


class DenoiserScoring(BaseScoring):
    metric_function: Union[str, Callable] = "rel l2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_metric_function()

    def set_metric_function(self):
        if not callable(self.metric_function):
            func = linalg_metric_func(self.metric_function, axis=-1)
            negative = bool(find_word_starting_with(self.metric_function, "neg", case_insensitive=True))
            sign = -1.0 if negative else 1.0

            def _metric_func(y_true, denoising_result):
                return sign * func(y_true, denoising_result.signal_denoised)

            self.metric_function = _metric_func


class DetectorScoring(DenoiserScoring):
    metric_function: Union[str, Callable] = "picks2.5 f0.5"

    def set_metric_function(self):
        if not callable(self.metric_function):
            picks = find_word_starting_with(self.metric_function, "pick", case_insensitive=True)
            func = binary_metric_func(self.metric_function)
            negative = bool(find_word_starting_with(self.metric_function, "neg", case_insensitive=True))
            sign = -1.0 if negative else 1.0

            def _metric_func(y_true, detector_result):
                if picks:
                    features = detector_result.picked_features
                    y_pred = np.empty(features.shape, dtype=object)
                    for ind in np.ndindex(*features.shape):
                        y_pred[ind] = features[ind][:, 0]
                else:
                    raise NotImplementedError("This metric has not been implemented yet")
                    # y_pred = ProjectIntervals(detector_result.detected_intervals)

                return sign * func(y_true, y_pred)

            self.metric_function = _metric_func


# class BayesTuner(BaseModel, extra=Extra.allow):
#     scoring_operator: Any
#     pbounds: dict[str, Union[tuple[float, float], tuple[int, int]]]
#     constraint: Any = None
#     random_state: Any = None
#     verbose: Any = 2
#     bounds_transformer: Any = None
#     allow_duplicate_points: Any = False
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.optimizer = BayesianOptimization(f=self.scoring_operator.score_function,
#                                               pbounds=self.pbounds,
#                                               random_state=self.random_state,
#                                               constraint=self.constraint,
#                                               verbose=self.verbose,
#                                               bounds_transformer=self.bounds_transformer,
#                                               allow_duplicate_points=self.allow_duplicate_points)
#
#     def tune(self, init_points=5, n_iter=25,
#              acquisition_function=None, acq=None, kappa=None,
#              kappa_decay=None, kappa_decay_delay=None, xi=None, **gp_params):
#
#         self.optimizer.maximize(init_points=init_points, n_iter=n_iter,
#                                 acquisition_function=acquisition_function,
#                                 acq=acq, kappa=kappa, kappa_decay=kappa_decay,
#                                 kappa_decay_delay=kappa_decay_delay, xi=xi, **gp_params)
#
#     @property
#     def best_params(self):
#         return self.optimizer.max
#
#     @property
#     def best_model(self):
#         self.scoring_operator.update_operator(**self.best_params['params'])
#         return self.scoring_operator.operator
#
#     def evaluate_best_model(self, x=None, y=None):
#         self.scoring_operator.update_operator(**self.best_params['params'])
#         return self.scoring_operator.evaluate_metric(x=x, y=y)
