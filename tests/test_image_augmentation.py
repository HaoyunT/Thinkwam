"""CPU regressions for shared LIBERO image/soft-heatmap augmentation."""

import pickle

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from thinkwam_p0.config import P0Config
from thinkwam_p0.data.augmentation import (
    ImageAugmentationConfig, ImageAugmenter, make_libero_subsets,
)
from thinkwam_p0.data.libero_online import OnlineLiberoDataset
from thinkwam_p1.config import P1Config
from thinkwam_p1.data.libero import P1LiberoDataset


class FixedSpatialRng:
    def __init__(self, first=(2, 2), second=(0, 0), both=False):
        self.flags = iter([0., 1., 1., 0., 1., 0. if both else 1.])
        self.shifts = iter([np.array(first), np.array(second)])

    def random(self):
        return next(self.flags)

    def uniform(self, lo, hi, size=None):
        return (lo + hi) / 2 if size is None else np.full(size, (lo + hi) / 2)

    def randint(self, lo, hi, size=None):
        return next(self.shifts)


def sample_with_heatmaps(layout="vhw"):
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    images[0, 2, 2] = 255
    images[1, 4, 4] = 255
    targets = np.zeros((2, 4, 4), dtype=np.float32)
    targets[0, 1, 1] = .8
    targets[1, 2, 2] = .6
    shape = {"vhw": (2, 4, 4), "vchw": (2, 1, 4, 4), "vhwc": (2, 4, 4, 1),
             "flat_views": (2, 16), "flat_tokens": (32, 1)}[layout]
    return dict(obs_images=images, future_images={7: images.copy(), 15: images.copy()},
                affordance_targets=torch.from_numpy(targets.reshape(shape)),
                actions=torch.ones(2, 7))


@pytest.mark.parametrize("layout", ["vhw", "vchw", "vhwc", "flat_views", "flat_tokens"])
def test_rgb_and_soft_heatmap_translation_match_each_view_and_resolution(layout):
    sample = sample_with_heatmaps(layout)
    original_target = sample["affordance_targets"].clone()
    augmenter = ImageAugmenter(ImageAugmentationConfig(probability=1.))
    augmenter.rng = FixedSpatialRng()
    out = augmenter(sample)
    assert np.argwhere(out["obs_images"][0, :, :, 0]).tolist() == [[4, 4]]
    assert np.argwhere(out["obs_images"][1, :, :, 0]).tolist() == [[6, 6]]
    target = out["affordance_targets"].reshape(2, 4, 4)
    assert target[0, 2, 2].item() == pytest.approx(.8)
    assert target[1, 3, 3].item() == pytest.approx(.6)
    assert torch.count_nonzero(target) == 2
    for images in out["future_images"].values():
        np.testing.assert_array_equal(images, out["obs_images"])
    torch.testing.assert_close(sample["affordance_targets"], original_target)
    assert out["actions"] is sample["actions"]


def test_fractional_heatmap_shift_preserves_soft_values_without_wrapping():
    sample = sample_with_heatmaps()
    augmenter = ImageAugmenter(ImageAugmentationConfig(probability=1.))
    augmenter.rng = FixedSpatialRng(first=(1, 0))
    out = augmenter(sample)["affordance_targets"]
    assert out[0, 1, 1].item() == pytest.approx(.4)
    assert out[0, 1, 2].item() == pytest.approx(.4)
    assert out[0].sum().item() == pytest.approx(.8)
    assert torch.count_nonzero(out[:, :, 0]) == 0


def test_two_translations_use_the_same_order_for_images_and_labels():
    sample = sample_with_heatmaps()
    augmenter = ImageAugmenter(ImageAugmentationConfig(probability=1.))
    augmenter.rng = FixedSpatialRng(first=(-4, 0), second=(4, 0), both=True)
    out = augmenter(sample)
    # The first view leaves the frame and must not reappear on the second shift.
    assert not out["obs_images"][0].any()
    assert not out["affordance_targets"][0].any()
    assert out["affordance_targets"][1, 2, 2].item() == pytest.approx(.6)


def test_appearance_changes_leave_heatmaps_unchanged_and_rng_is_seeded():
    cfg = ImageAugmentationConfig(probability=1., illumination_probability=1.,
                                  texture_probability=1., noise_probability=1.,
                                  camera_probability=0., object_position_probability=0.)
    sample = sample_with_heatmaps()
    augmenter = pickle.loads(pickle.dumps(ImageAugmenter(cfg)))
    np.random.seed(7)
    first = augmenter(sample)
    np.random.seed(7)
    second = augmenter(sample)
    np.testing.assert_array_equal(first["obs_images"], second["obs_images"])
    assert first["affordance_targets"] is sample["affordance_targets"]
    assert not np.array_equal(first["obs_images"], sample["obs_images"])


@pytest.mark.parametrize("stage", ["p0", "p1"])
def test_both_libero_stages_augment_training_and_keep_validation_raw(tmp_path, stage):
    with h5py.File(tmp_path / "task.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        for camera in ("agentview_rgb", "eye_in_hand_rgb"):
            obs.create_dataset(camera, data=np.full((8, 8, 8, 3), 100, dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((8, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((8, 2), dtype=np.float32))
        obs.create_dataset("affordance", data=np.ones((8, 2, 4, 4), dtype=np.float32))
    cls, config_cls = (OnlineLiberoDataset, P0Config) if stage == "p0" else (P1LiberoDataset, P1Config)
    dataset = cls(str(tmp_path), action_horizon=2, horizons=(1, 2, 3))
    config = config_cls(image_augmentation_probability=1., image_illumination_probability=1.,
                        image_texture_probability=0., image_camera_probability=0.,
                        image_object_position_probability=0., image_noise_probability=0.)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7)).tolist()
    train, val = make_libero_subsets(dataset, indices[:-1], indices[-1:], config)
    raw_train = dataset[indices[0]]["obs_images"]
    # Exercise DataLoader's batched __getitems__, not only direct indexing.
    batch = next(iter(DataLoader(train, batch_size=2, collate_fn=lambda items: items)))
    assert not np.array_equal(batch[0]["obs_images"], raw_train)
    np.testing.assert_array_equal(val[0]["obs_images"], dataset[indices[-1]]["obs_images"])
    assert train.indices == indices[:-1] and val.indices == indices[-1:]


@pytest.mark.parametrize("field", ["plus_dataset_root", "plus_augmentations", "plus_mixture_fraction",
                                   "task_sampling_enabled", "robocasa_event_sampling",
                                   "phase_enabled", "vjepa_trainable"])
def test_removed_training_options_fail_instead_of_being_silently_ignored(field):
    with pytest.raises(TypeError, match=field):
        P1Config(**{field: True})


def test_paper_defaults_and_removed_warmup_stage():
    for config in (P0Config(), P1Config()):
        assert config.action_horizon == 16 and config.sigreg_loss_weight == 0.
        assert config.image_augmentations and config.image_augmentation_probability == .5
        assert config.online_vjepa and config.qwen_lora
    with pytest.raises(ValueError, match="p1_stage"):
        P1Config(p1_stage="warmup")
