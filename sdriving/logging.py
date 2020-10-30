import atexit
import json
import os
import os.path as osp
import time

import horovod.torch as hvd
import joblib
import numpy as np
import torch
import wandb

from sdriving.agents.utils import (
    hvd_scalar_statistics,
    hvd_scalar_statistics_with_min_max,
)

import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from fire import Fire


def plot_experiment_logs(fname: str, paths, tags=None):
    if tags is None:
        tags = paths
    dfs = [pd.read_csv(path) for path in paths]
    for tag, df in zip(tags, dfs):
        for col in df.columns:
            df[col] = [float(x[7:-1]) for x in df[col]]
        df["Tag"] = [tag] * df.shape[0]
        df["Epoch"] = list(range(df.shape[0]))
    df = pd.concat(dfs)

    ncol = 3
    nrow = math.ceil((len(df.columns) - 1) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 15))
    i = 0
    for col in df.columns:
        if col == "Epoch" or col == "Tag":
            continue
        sns.lineplot(
            x="Epoch",
            y=col,
            data=df,
            ax=axs[i // ncol][i % ncol],
            hue="Tag",
        )
        i += 1

    plt.tight_layout()
    plt.savefig(fname)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}
        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attr = ";".join(attr)
    return f"\x1b[{attr}m{string}\x1b[0m"


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
        self, output_dir=None, output_fname="progress.csv", exp_name=None
    ):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if hvd.rank() == 0:
            self.output_dir = (
                output_dir or f"/tmp/experiments/{int(time.time())}"
            )
            if osp.exists(self.output_dir):
                print(
                    f"Warning: Log dir {self.output_dir} already exists!"
                    f" Storing info there anyway."
                )
            else:
                os.makedirs(self.output_dir, exist_ok=True)
            file_name = osp.join(self.output_dir, output_fname)
            self.first_row = not osp.exists(file_name)
            self.output_file = open(file_name, "a")
            atexit.register(self.output_file.close)
            print(
                colorize(
                    f"Logging data to {self.output_file.name}",
                    "green",
                    bold=True,
                )
            )
        else:
            self.output_dir = None
            self.output_file = None
            self.first_row = True
        self.first_log = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        if hvd.rank() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_log:
            self.log_headers.append(key)
        assert key not in self.log_current_row, AssertionError(
            f"You already set {key} this iteration."
            f" Maybe you forgot to call dump_tabular()"
        )
        self.log_current_row[key] = val

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if hvd.rank() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = "%" + "%d" % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes, flush=True)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write(",".join(self.log_headers) + "\n")
                self.output_file.write(",".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False
        self.first_log = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
        if hvd.rank() == 0:
            wandb.log(kwargs)

    def log_tabular(
        self, key, val=None, with_min_and_max=False, average_only=False
    ):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            vals = torch.as_tensor(vals).float()
            if with_min_and_max:
                stats = hvd_scalar_statistics_with_min_max(vals)
            else:
                stats = hvd_scalar_statistics(vals)

            super().log_tabular(
                key if average_only else "Average" + key, stats[0]
            )
            if not (average_only):
                super().log_tabular("Std" + key, stats[1])
            if with_min_and_max:
                super().log_tabular("Max" + key, stats[3])
                super().log_tabular("Min" + key, stats[2])
        self.epoch_dict[key] = []


if __name__ == "__main__":
    Fire({"plot_experiment_logs": plot_experiment_logs})
