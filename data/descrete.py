from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DescreteEncoder:
    def __init__(self, duration: int = 1440, step_size: int = 10):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size
        self.index_to_acts = {}
        self.acts_to_index = {}

    def encode(self, data: pd.DataFrame):
        # Create mappings from activity to index and vice versa
        self.index_to_acts = {i: a for i, a in enumerate(data.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}

        return DiscreteEncoded(
            data,
            duration=self.duration,
            step_size=self.step_size,
            class_map=self.acts_to_index,
        )

    def decode(self, encoded_image_grid) -> pd.DataFrame:
        if isinstance(encoded_image_grid, torch.Tensor):
            encoded_image_grid = encoded_image_grid.numpy()

        decoded = []
        for pid in range(encoded_image_grid.shape[0]):
            sequence = encoded_image_grid[pid]
            current_act = None
            act_start = None

            for time_step, act_index in enumerate(sequence):
                # If the activity changes or it's the end of the day, record the activity
                if act_index != current_act and current_act is not None:
                    act_end = time_step * self.step_size
                    decoded.append(
                        {
                            "pid": pid,
                            "act": self.index_to_acts[current_act],
                            "start": act_start,
                            "end": act_end,
                        }
                    )
                    act_start = time_step * self.step_size
                # If the activity changes, update the current activity
                if act_index != current_act:
                    current_act = act_index
                    act_start = time_step * self.step_size

            # Add the last activity of the day if the day ended with an activity
            if current_act is not None and act_start is not None:
                decoded.append(
                    {
                        "pid": pid,
                        "act": self.index_to_acts[current_act],
                        "start": act_start,
                        "end": self.duration,
                    }
                )

        return pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])


class DiscreteEncoded(Dataset):
    def __init__(
        self, data: pd.DataFrame, duration: int, step_size: int, class_map: dict
    ):
        """Torch Dataset for descretised sequence data.

        Args:
            data (Tensor): Population of sequences.
        """
        data = data.copy()
        data.act = data.act.map(class_map)
        self.encodings = data.act.nunique()
        # calc weightings
        weights = data.groupby("act", observed=True).duration.sum().to_dict()
        weights = np.array([weights[k] for k in range(len(weights))])
        self.encoding_weights = torch.from_numpy(1 / weights).float()
        self.encoded = discretise_population(
            data, duration=duration, step_size=step_size
        )
        self.mask = torch.ones((1, self.encoded.shape[-1]))
        self.size = len(self.encoded)

    def shape(self):
        return self.encoded[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.encoded[idx]
        return (sample, self.mask), (sample, self.mask)


def discretise_population(
    data: pd.DataFrame, duration: int, step_size: int
) -> torch.Tensor:
    """Convert given population of activity traces into vector [N, L] of classes.
    N is the population size.
    l is time steps.

    Args:
        data (pd.DataFrame): _description_
        duration (int): _description_
        step_size (int): _description_

    Returns:
        torch.tensor: [N, L]
    """
    persons = data.pid.nunique()
    steps = duration // step_size
    encoded = np.zeros((persons, steps), dtype=np.int8)

    for pid, (_, trace) in enumerate(data.groupby("pid")):
        trace_encoding = discretise_trace(
            acts=trace.act, starts=trace.start, ends=trace.end, length=duration
        )
        trace_encoding = down_sample(trace_encoding, step_size)
        encoded[pid] = trace_encoding  # [N, L]
    return torch.from_numpy(encoded)


def discretise_trace(
    acts: Iterable[str], starts: Iterable[int], ends: Iterable[int], length: int
) -> np.ndarray:
    """Create categorical encoding from ranges with step of 1.

    Args:
        acts (Iterable[str]): _description_
        starts (Iterable[int]): _description_
        ends (Iterable[int]): _description_
        length (int): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((length), dtype=np.int8)
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = act
    return encoding


def down_sample(array: np.ndarray, step: int) -> np.ndarray:
    """Down-sample by steppiong through given array.
    todo:
    Methodology will down sample based on first classification.
    If we are down sampling a lot (for example from minutes to hours),
    we would be better of, sampling based on majority class.

    Args:
        array (np.array): _description_
        step (int): _description_

    Returns:
        np.array: _description_
    """
    return array[::step]
