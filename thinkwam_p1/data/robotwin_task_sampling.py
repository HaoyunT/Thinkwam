"""Equal task probability for the 50 RoboTwin Clean training tasks.

Within the already selected training split, each window has weight
``1 / (50 * windows_in_its_task)``. Sampling uses replacement: task probabilities
are equal, while individual finite batches need not contain equal task counts.
"""

from __future__ import annotations

import bisect
from collections import Counter
import math
import re

import torch
from torch.utils.data import Sampler, Subset


ROBOTWIN_TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot", "move_can_pot",
    "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left",
    "place_a2b_right", "place_bread_basket", "place_bread_skillet", "place_burger_fries",
    "place_can_basket", "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad", "place_object_basket",
    "place_object_scale", "place_object_stand", "place_phone_stand", "place_shoe",
    "press_stapler", "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
    "scan_object", "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three",
    "stack_blocks_two", "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)


def _task_name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower().replace("&", " and ")).strip("_")


def _episode_tasks(episodes) -> list[str]:
    names = [_task_name(episode.get("task_name") or episode.get("task") or "")
             for episode in episodes]
    if all(names):
        return names
    # The existing Clean RGB export stores 50 demos per task in this fixed
    # order, but no task field. Language is a per-demo paraphrase, not a task ID.
    # Restrict this fallback to that complete export; partial/new exports must
    # carry explicit task names instead of silently inferring missing IDs.
    episode_ids = [int(episode["episode_index"]) for episode in episodes]
    if len(episode_ids) != 2500 or set(episode_ids) != set(range(2500)):
        raise ValueError("RoboTwin episodes need explicit task names or the complete 0..2499 Clean export")
    inferred = [ROBOTWIN_TASKS[episode_id // 50] for episode_id in episode_ids]
    if any(name and name != expected for name, expected in zip(names, inferred)):
        raise ValueError("Explicit RoboTwin task metadata conflicts with the Clean export order")
    return inferred


def _split_task_ids(dataset) -> tuple[str, ...]:
    indices = list(range(len(dataset)))
    base = dataset
    while isinstance(base, Subset):
        indices = [int(base.indices[index]) for index in indices]
        base = base.dataset
    if not hasattr(base, "episodes") or not hasattr(base, "ends"):
        raise TypeError("Uniform RoboTwin sampling requires OnlineRobotWinDataset episode/window metadata")
    ends = list(map(int, base.ends))
    if (len(ends) != len(base.episodes) or not ends or ends[-1] != len(base)
            or ends[0] < 0 or any(left > right for left, right in zip(ends, ends[1:]))):
        raise ValueError("Invalid RoboTwin cumulative episode/window metadata")
    names = _episode_tasks(base.episodes)
    result = []
    for index in indices:
        if index < 0:
            index += len(base)
        if not 0 <= index < len(base):
            raise IndexError(f"RoboTwin subset window is out of range: {index}")
        result.append(names[bisect.bisect_right(ends, index)])
    return tuple(result)


class UniformRobotWinTaskSampler(Sampler[int]):
    """Task-uniform training draws, deterministically partitioned across DDP ranks."""

    def __init__(self, dataset_subset, *, num_replicas: int = 1, rank: int = 0, seed: int = 0):
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid RoboTwin sampler rank/replica count")
        self.task_ids = _split_task_ids(dataset_subset)
        self.task_counts = Counter(self.task_ids)
        expected = set(ROBOTWIN_TASKS)
        if set(self.task_counts) != expected:
            raise ValueError(
                "The RoboTwin training split must contain all 50 tasks; "
                f"missing={sorted(expected - self.task_counts.keys())}, "
                f"unexpected={sorted(self.task_counts.keys() - expected)}"
            )
        self.weights = torch.tensor(
            [1.0 / (50 * self.task_counts[task]) for task in self.task_ids],
            dtype=torch.double,
        )
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(self.task_ids) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(draws[self.rank:self.total_size:self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples
