from pydantic import BaseModel, Field
from typing import Any, Union, List, Dict, Callable
import numpy as np
import colorednoise as cn
from itertools import product
from tqdm.notebook import tqdm


class TraceAugmenter(BaseModel):
    traces: List[Any]
    reference_outputs: Union[List[Any], None] = None
    attributes: Union[List[Dict], None] = None
    reference_level: Union[List[float], None] = None
    random_noise_levels: Union[List[float], List[List[float]]] = [0.5]
    random_noise_colors: Union[List[float], List[List[float]]] = [0.0]
    random_noise_repeats: int = 2
    product_levels_and_colors: bool = False
    reference_noise: List[Any] = []
    reference_noise_levels: Union[List[float], List[List[float]]] = [0.5]
    trace_dropout_rate: float = Field(0.0, ge=0, lt=1.0)
    # shuffle_traces: bool = False
    # shuffle_axis: Union[int, None] = None
    progress_bar: bool = False
    callbacks: List[Callable] = []
    noise_reference_name: str = "reference"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reference_outputs = self.reference_outputs or self.traces
        self.attributes = self.attributes or [{}] * len(self.traces)
        self.reference_level = self.reference_level or [1] * len(self.traces)

        self._set_nested_list('random_noise_levels', len(self.traces))
        self._set_nested_list('random_noise_colors', len(self.traces))
        self._set_nested_list('reference_noise_levels', len(self.reference_noise))

        assert len(self.attributes) == len(self.traces)
        assert len(self.reference_outputs) == len(self.traces)
        assert len(self.random_noise_levels) == len(self.traces)
        assert len(self.random_noise_colors) == len(self.traces)
        assert len(self.reference_noise_levels) == len(self.reference_noise)
        if not self.product_levels_and_colors:
            assert (sum(self._len_of_nested_list(self.random_noise_levels)) ==
                    sum(self._len_of_nested_list(self.random_noise_colors)))

    def _set_nested_list(self, attr, ref_len):
        val = getattr(self, attr)
        if (len(val) == 0) or (not isinstance(val[0], list)):
            setattr(self, attr, [val] * ref_len)

    @staticmethod
    def _len_of_nested_list(nested_list: List[List[Any]]):
        return [len(nli) for nli in nested_list]

    @property
    def get_total_iterations(self):
        if self.product_levels_and_colors:
            levels_len = self._len_of_nested_list(self.random_noise_levels)
            colors_len = self._len_of_nested_list(self.random_noise_colors)
            random_len = sum([l1 * l2 for l1, l2 in zip(levels_len, colors_len)]) * self.random_noise_repeats
        else:
            random_len = sum(self._len_of_nested_list(self.random_noise_levels)) * self.random_noise_repeats
        reference_len = sum(self._len_of_nested_list(self.reference_noise_levels)) * len(self.traces)
        return random_len + reference_len

    def __len__(self):
        return self.get_total_iterations

    @staticmethod
    def generate_noise(shape, level, color):
        if isinstance(color, (float, int)):
            if color:
                noise = cn.powerlaw_psd_gaussian(color, shape)
            else:
                noise = np.random.randn(*shape)
        elif isinstance(color, np.ndarray):
            noise = np.zeros(shape)
            for ind in np.ndindex(color.shape):
                noise[ind] = TraceAugmenter.generate_noise(noise[ind].shape, level, color[ind])
        else:
            raise ValueError

        return level * noise

    def shuffle_dropout(self, traces, output):
        # if self.shuffle_traces:
        #     raise NotImplementedError('Shuffling has not been implemented yet')
        #     shuffle_ind = np.arange(traces.shape[self.shuffle_axis])
        #     np.random.shuffle(shuffle_ind)
        #     traces = np.take_along_axis(traces, indices=shuffle_ind, axis=self.shuffle_axis)
        #     if output.shape[:-1] == traces.shape[:-1]:
        #         output = np.take_along_axis(output, indices=shuffle_ind, axis=self.shuffle_axis)
        dropout_c = np.random.choice([0.0, 1.0], size=traces.shape[:-1] + (1,),
                                     p=[self.trace_dropout_rate, 1 - self.trace_dropout_rate])
        return dropout_c * traces, output

    def __call__(self):
        repeats = self.random_noise_repeats
        ref_name = self.noise_reference_name
        rand_prod = self.product_levels_and_colors
        cntr = 0
        zipped = zip(enumerate(self.traces), self.reference_outputs, self.reference_level,
                     self.attributes, self.random_noise_levels, self.random_noise_colors)
        with (tqdm(desc="Augmenter", total=len(self), display=self.progress_bar) as pbar):
            for ((i, tr), out, r_l, attrs, nls_rand, ncs) in zipped:
                traces = tr.copy()
                output = out.copy()

                random_noise_opts = (product(nls_rand * repeats, ncs) if rand_prod
                                     else zip(nls_rand * repeats, ncs * repeats))

                for n_l, n_c in random_noise_opts:
                    noise = self.generate_noise(traces.shape, n_l * r_l, n_c)
                    cntr = self.report_to_callbacks(cntr, trace=i, noise_level=n_l, noise_color=n_c,
                                                    noise_name='random')
                    data_in, data_out = self.shuffle_dropout(traces + noise, output)
                    pbar.update()
                    yield data_in, data_out, attrs

                for n_ref, nls_ref in zip(self.reference_noise, self.reference_noise_levels):
                    for n_l in nls_ref:
                        noise = n_ref / np.std(n_ref) * n_l * r_l

                        cntr = self.report_to_callbacks(cntr, trace=i, noise_level=n_l, noise_color=None,
                                                        noise_name=ref_name)
                        data_in, data_out = self.shuffle_dropout(traces + noise, output)
                        pbar.update()
                        yield data_in, data_out, attrs

    def report_to_callbacks(self, cntr, **kwargs):
        cntr += 1
        kwargs.setdefault("counter", cntr)
        for cb in self.callbacks:
            cb(**kwargs)
        return cntr


class Recorder:  # records auxiliary params such as iter number
    container: Dict

    def __init__(self):
        self.restart()

    def __call__(self, **kwargs):
        for kw, val in kwargs.items():
            if self.container.get(kw, None) is None:
                self.container[kw] = [val]
            else:
                self.container[kw].append(val)

    def restart(self):
        self.container = {}
