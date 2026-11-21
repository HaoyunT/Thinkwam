"""CPU metadata-only tests: constructing the sampler must not decode RGB."""

import bisect
from collections import defaultdict
import itertools
import unittest

import torch
from torch.utils.data import Dataset, Subset

from thinkwam_p1.data.robotwin_task_sampling import ROBOTWIN_TASKS, UniformRobotWinTaskSampler


class WindowMetadata(Dataset):
    def __init__(self, counts, tasks=None):
        self.episodes = [dict(episode_index=index, language=f"demo paraphrase {index}")
                         for index in range(len(counts))]
        if tasks is not None:
            for episode, task in zip(self.episodes, tasks):
                episode["task_name"] = task
        self.ends = list(itertools.accumulate(counts))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        raise AssertionError("A metadata-only sampler must not read or decode a sample")


class RobotWinTaskSamplingTest(unittest.TestCase):
    def setUp(self):
        self.base = WindowMetadata([1 + index % 7 for index in range(50)], ROBOTWIN_TASKS)

    def test_nested_split_mapping_gives_every_task_exactly_one_fiftieth_mass(self):
        selected = []
        start = 0
        for end in self.base.ends:
            selected.extend(range(start, end, 2))
            start = end
        subset = Subset(self.base, selected)
        order = list(reversed(range(len(subset))))
        nested = Subset(subset, order)
        sampler = UniformRobotWinTaskSampler(nested)
        masses = defaultdict(float)
        for local_index, (task, weight) in enumerate(zip(sampler.task_ids, sampler.weights)):
            original_index = selected[order[local_index]]
            expected = ROBOTWIN_TASKS[bisect.bisect_right(self.base.ends, original_index)]
            self.assertEqual(task, expected)
            masses[task] += float(weight)
        self.assertEqual(set(masses), set(ROBOTWIN_TASKS))
        for mass in masses.values():
            self.assertAlmostEqual(mass, 1.0 / 50, places=14)

    def test_complete_clean_export_uses_episode_ids_not_language(self):
        base = WindowMetadata([1] * 2500)
        base.episodes.reverse()
        sampler = UniformRobotWinTaskSampler(base)
        self.assertEqual(sampler.task_ids[0], ROBOTWIN_TASKS[-1])
        self.assertEqual(sampler.task_ids[-1], ROBOTWIN_TASKS[0])
        self.assertEqual(set(sampler.task_counts.values()), {50})
        base.episodes[0]["task_name"] = ROBOTWIN_TASKS[0]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            UniformRobotWinTaskSampler(base)

    def test_missing_unknown_and_unresolvable_tasks_are_rejected(self):
        missing_last_task = Subset(self.base, range(self.base.ends[-2]))
        with self.assertRaisesRegex(ValueError, "all 50 tasks"):
            UniformRobotWinTaskSampler(missing_last_task)
        self.base.episodes[0]["task_name"] = "not_a_training_task"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            UniformRobotWinTaskSampler(self.base)
        with self.assertRaisesRegex(ValueError, "explicit task names"):
            UniformRobotWinTaskSampler(WindowMetadata([1] * 50))

    def test_epoch_and_seed_make_draws_repeatable(self):
        sampler = UniformRobotWinTaskSampler(self.base, seed=7)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        identical = UniformRobotWinTaskSampler(self.base, seed=7)
        identical.set_epoch(1)
        self.assertEqual(second, list(identical))
        self.assertTrue(all(0 <= index < len(self.base) for index in second))

    def test_ddp_ranks_partition_the_same_global_draw_without_losing_positions(self):
        samplers = [UniformRobotWinTaskSampler(self.base, num_replicas=4, rank=rank, seed=19)
                    for rank in range(4)]
        for sampler in samplers:
            sampler.set_epoch(3)
        ranks = [list(sampler) for sampler in samplers]
        self.assertTrue(all(len(draws) == len(samplers[0]) for draws in ranks))
        combined = [value for column in zip(*ranks) for value in column]
        expected = torch.multinomial(
            samplers[0].weights, samplers[0].total_size, replacement=True,
            generator=torch.Generator().manual_seed(22),
        ).tolist()
        self.assertEqual(combined, expected)


if __name__ == "__main__":
    unittest.main()
