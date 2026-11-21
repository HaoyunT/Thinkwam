"""Deterministic holdout of complete demonstration episodes."""

import torch


def episode_holdout_indices(index, *, eval_fraction: float, seed: int):
    if not 0 < eval_fraction < 1:
        raise ValueError("episode holdout requires 0 < eval_fraction < 1")
    episodes = sorted({episode for episode, _frame in index})
    if len(episodes) < 2:
        raise ValueError("episode holdout requires at least two nonempty episodes")
    order = torch.randperm(
        len(episodes), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    count = min(len(episodes) - 1, max(1, int(len(episodes) * eval_fraction)))
    held_out = {episodes[i] for i in order[:count]}
    train, validation = [], []
    for i, (episode, _frame) in enumerate(index):
        (validation if episode in held_out else train).append(i)
    return train, validation
