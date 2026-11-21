from __future__ import annotations

import json

import numpy as np

from scripts.build_robotwin_vjepa_cache import _load_jsonl_label_maps
from thinkwam_p0.data.robotwin import RobotWinEpisode, load_affordance_episode
from thinkwam_p1.data.robotwin_affordance import deva_normalize_and_pool


def test_deva_pool_is_per_frame_scale_invariant() -> None:
    values = np.zeros((2, 4, 6), dtype=np.float32)
    values[0, 1, 2] = 2.0
    values[1, 2, 4] = 4.0
    pooled = deva_normalize_and_pool(values, (2, 3))
    pooled_scaled = deva_normalize_and_pool(values * 7.0, (2, 3))
    np.testing.assert_allclose(pooled, pooled_scaled, rtol=0.0, atol=1e-6)
    assert pooled.dtype == np.float16
    assert float(pooled.max()) <= 1.0


def test_robotwin_affordance_loader_accepts_simulator_camera_names(tmp_path) -> None:
    directory = tmp_path / "clean" / "7"
    directory.mkdir(parents=True)
    for name in ("head_camera", "left_camera", "right_camera"):
        np.savez_compressed(directory / f"{name}.npz", **{name: np.zeros((3, 4, 5), dtype=np.float16)})
    episode = RobotWinEpisode(
        root=tmp_path,
        variant="clean",
        episode_index=7,
        length=3,
        data_path=tmp_path / "data.parquet",
        data_from=0,
        data_to=3,
        task_index=0,
        language="task",
        video_paths=(),
        video_starts=(),
    )
    loaded = load_affordance_episode(
        tmp_path,
        episode,
        (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
    )
    assert loaded is not None
    assert set(loaded) == {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }


def test_jsonl_loader_rasterizes_one_frame_and_keeps_missing_frames_zero(tmp_path) -> None:
    directory = tmp_path / "clean" / "3"
    directory.mkdir(parents=True)
    camera_config = {
        name: {
            "height": 8,
            "width": 8,
            "intrinsic_cv": [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
            "extrinsic_cv": np.eye(4).tolist(),
        }
        for name in ("head_camera", "left_camera", "right_camera")
    }
    (directory / "affordance.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 0,
                "contacts_world": [[0.0, 0.0, 1.0]],
                "cameras": camera_config,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = RobotWinEpisode(
        root=tmp_path,
        variant="clean",
        episode_index=3,
        length=2,
        data_path=tmp_path / "data.parquet",
        data_from=0,
        data_to=2,
        task_index=0,
        language="task",
        video_paths=(),
        video_starts=(),
    )
    maps = _load_jsonl_label_maps(
        tmp_path,
        episode,
        ("observation.images.cam_high", "observation.images.cam_left_wrist"),
        map_size=2,
    )
    assert maps is not None
    assert maps.shape == (2, 2, 2, 2)
    assert float(maps[0].max()) > 0.0
    np.testing.assert_array_equal(maps[1], 0.0)
