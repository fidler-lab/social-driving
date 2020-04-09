from typing import List, Tuple, Union

import numpy as np
import torch


class TrafficSignal:
    def __init__(
        self,
        val: Union[List[int], Tuple[int], int] = 2,
        start_signal: int = 0,
        times: Union[List[int], Tuple[int]] = [10] * 2,
        name: str = "signal",
        colors: Union[List[str], Tuple[str]] = ["g", "r"],
    ):
        """
        The value returned by the signal is a floating point number
        To simulate a normal traffic light with 3 colors - R, G, Y
        the argument `val` should be [0.0, 0.5, 1.0, 0.5]
        """
        if isinstance(val, int):
            self.n = val
            self.values = torch.linspace(0, 1, val)
        else:
            self.n = len(val)
            self.values = torch.as_tensor(val)
        assert len(times) == self.n
        self.times = times
        self.name = name

        self.cur_light = start_signal
        self.time_past = 0
        self.start_signal = start_signal

        self.colors = {
            val.item(): color for val, color in zip(self.values, colors)
        }

    def reset(self):
        self.time_past = 0
        self.cur_light = self.start_signal

    def update_lights(self, time: int = 1):
        self.time_past += time
        if self.time_past >= self.times[self.cur_light]:
            self.time_past %= self.times[self.cur_light]
            self.cur_light = (self.cur_light + 1) % self.n

    def __repr__(self):
        switch = self.times[self.cur_light] - self.time_past
        ret = f"Traffic Signal: {self.name} | {self.n} |"
        ret += f" {self.get_value()} | Switches in {switch}"
        return ret

    def get_value(self):
        return self.values[self.cur_light]

    def get_color(self):
        return self.colors[self.get_value().item()]
