from pydantic import BaseModel, Extra
from typing import Union, Any, Callable
from cats.metrics import metric_func
from bayes_opt import BayesianOptimization
import numpy as np


class BayesTuner(BaseModel, extra=Extra.allow):
    scoring_operator: Any
    pbounds: dict[str, Union[tuple[float, float], tuple[int, int]]]
    constraint: Any = None
    random_state: Any = None
    verbose: Any = 2
    bounds_transformer: Any = None
    allow_duplicate_points: Any = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = BayesianOptimization(f=self.scoring_operator.score_function,
                                              pbounds=self.pbounds,
                                              random_state=self.random_state,
                                              constraint=self.constraint,
                                              verbose=self.verbose,
                                              bounds_transformer=self.bounds_transformer,
                                              allow_duplicate_points=self.allow_duplicate_points)

    def tune(self, init_points=5, n_iter=25,
             acquisition_function=None, acq=None, kappa=None,
             kappa_decay=None, kappa_decay_delay=None, xi=None, **gp_params):

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter,
                                acquisition_function=acquisition_function,
                                acq=acq, kappa=kappa, kappa_decay=kappa_decay,
                                kappa_decay_delay=kappa_decay_delay, xi=xi, **gp_params)

    @property
    def best_params(self):
        return self.optimizer.max

    @property
    def best_model(self):
        self.scoring_operator.update_operator(**self.best_params['params'])
        return self.scoring_operator.operator


class BaseScoring(BaseModel, extra=Extra.allow):
    operator: Any
    reference_data: Any
    noisy_data_generator: Callable
    metric_function: Callable
    time_coefficient: float = 0.0
    time_power: Union[float, int] = 2
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

    def score_function(self, **kwargs):
        self.update_operator(**kwargs)
        score = 0.0
        n_times = 0
        for data_i in self.noisy_data_generator():
            if self.prepare_operator is not None:
                self.prepare_operator(self.operator, data_i)

            operator_result = self.operator * data_i
            score_i = self.metric_function(self.reference_data, operator_result)

            if self.time_coefficient:
                elapsed = operator_result.history.history['Total']
                score_i *= np.exp(- self.time_coefficient * elapsed**self.time_power)
            n_times += 1
            score += score_i

        return score / n_times


class DenoiserScoring(BaseScoring):
    metric_function: Union[str, Callable] = "rel acc l2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_metric_function()

    def set_metric_function(self):
        func = metric_func(self.metric_function, axis=-1)

        def _metric_func(y_true, denoising_result):
            return func(y_true, denoising_result.signal_denoised)

        self.metric_function = _metric_func


class DetectorScoring(BaseScoring):
    pass