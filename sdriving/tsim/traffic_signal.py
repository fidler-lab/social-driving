from typing import List

import torch


class TrafficSignal:
    def __init__(
        self,
        val: torch.Tensor,
        start_signal: int,
        times: torch.Tensor,
        name: str = "signal",
        colors: List[str] = ["g", "r"],
    ):
        self.n = val.size(0)
        self.values = val
        assert times.size(0) == self.n
        self.times = times
        self.name = name

        self.cur_light = start_signal
        self.time_past = torch.zeros(1)
        self.start_signal = start_signal

        self.colors = {
            val.item(): color for val, color in zip(self.values, colors)
        }

    def reset(self):
        self.time_past = torch.zeros(1)
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
