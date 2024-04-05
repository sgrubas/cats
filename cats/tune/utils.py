from pydantic import BaseModel, Field
from typing import Any, Union, List, Dict, Callable
import numpy as np
import colorednoise as cn
from itertools import product
from tqdm.notebook import tqdm


class TraceAugmenter(BaseModel):
    traces: List[Any]
    reference_outputs: List[Any] = None
    attributes: List[Dict] = None
    noise_levels: List[Union[float, Any]]
    reference_level: List[float]
    noise_colors: List[Union[float, Any]] = [0.0]
    noise_random_repeats: int = 2
    noise_references: List[Any] = []
    trace_dropout_rate: float = Field(0.0, ge=0, lt=1.0)
    shuffle_traces: bool = False
    progress_bar: bool = False
    callbacks: List[Callable] = []
    noise_reference_name: str = "reference"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reference_outputs = self.reference_outputs or self.traces
        self.attributes = self.attributes or [{}] * len(self.traces)
        assert len(self.attributes) == len(self.traces)
        assert len(self.reference_outputs) == len(self.traces)

    @property
    def get_total_iterations(self):
        return (len(self.traces) * len(self.noise_levels) *
                (len(self.noise_colors) * self.noise_random_repeats + len(self.noise_references)))

    def __len__(self):
        return self.get_total_iterations

    def generate_noise(self, shape, level, color):
        if isinstance(color, (float, int)):
            if color:
                noise = cn.powerlaw_psd_gaussian(color, shape)
            else:
                noise = np.random.randn(*shape)
        elif isinstance(color, np.ndarray):
            noise = np.zeros(shape)
            for ind in np.ndindex(color.shape):
                noise[ind] = self.generate_noise(noise[ind].shape, level, color[ind])
        else:
            raise ValueError

        return level * noise

    def shuffle_dropout(self, traces):
        if self.shuffle_traces:
            np.random.shuffle(traces)
        dropout_c = np.random.choice([0.0, 1.0], size=traces.shape[:-1] + (1,),
                                     p=[self.trace_dropout_rate, 1 - self.trace_dropout_rate])
        return dropout_c * traces

    def __call__(self):
        cntr = 0
        with (tqdm(desc="Augmenter", total=len(self), display=self.progress_bar) as pbar):
            for (tr, out, r_l, attrs), n_l in product(zip(self.traces, self.reference_outputs,
                                                          self.reference_level, self.attributes),
                                                      self.noise_levels):
                traces = tr.copy()

                for n_c in self.noise_colors * self.noise_random_repeats:
                    noise = self.generate_noise(traces.shape, n_l * r_l, n_c)

                    cntr = self.report_to_callbacks(cntr, noise_level=n_l, noise_color=n_c)
                    pbar.update()
                    yield self.shuffle_dropout(traces) + noise, out, attrs

                for n_ref in self.noise_references:
                    noise = n_ref / np.std(n_ref) * n_l * r_l

                    cntr = self.report_to_callbacks(cntr, noise_level=n_l, noise_color=self.noise_reference_name)
                    pbar.update()
                    yield self.shuffle_dropout(traces) + noise, out, attrs

    def report_to_callbacks(self, cntr, **kwargs):
        cntr += 1
        kwargs.setdefault("counter", cntr)
        for cb in self.callbacks:
            cb(**kwargs)
        return cntr
