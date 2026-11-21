"""Standard LIBERO inputs shared by P1 evaluation and cache diagnostics.

Uses the training image, proprioception and raw-gripper conventions.
"""
from pathlib import Path
import json
import os

import numpy as np

LIBERO_ROOT = Path(os.environ.get("THINKWAM_LIBERO_ROOT", "/data/thinkwam/src/LIBERO"))
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = dict(zip(LIBERO_SUITES, (220, 280, 300, 520)))


def max_steps_for_suite(suite):
    if suite not in MAX_STEPS:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    return MAX_STEPS[suite]


def get_proprio(obs, dim=8):
    import torch
    from robosuite.utils.transform_utils import quat2axisangle
    pos = np.asarray(obs["robot0_eef_pos"]).reshape(-1)
    quat = np.asarray(obs["robot0_eef_quat"]).reshape(-1).copy()
    grip = np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)
    value = np.concatenate([pos, quat2axisangle(quat), grip]).astype(np.float32)
    if dim != 8 or value.shape != (8,):
        raise ValueError(f"Expected position(3), axis-angle(3), gripper(2), got {value.shape}")
    return torch.from_numpy(value)


def encode_obs(encoder, obs, device, token_count):
    from thinkwam_p0.data.libero_inputs import online_cache_features
    return online_cache_features(encoder, obs, device, token_count)

def ensure_libero_config() -> None:
    """Bind LIBERO's path config to this checkout, including a separate Plus root."""
    import hashlib
    import tempfile
    import yaml

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    package_root = LIBERO_ROOT.resolve() / "libero" / "libero"
    root_id = hashlib.sha256(str(package_root).encode()).hexdigest()[:16]
    default_dir = Path(tempfile.gettempdir()) / "thinkwam_libero_config" / root_id
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH") or
                      os.environ.get("THINKWAM_LIBERO_CONFIG_DIR", str(default_dir)))
    config_file = config_dir / "config.yaml"
    expected = {"benchmark_root": str(package_root), "bddl_files": str(package_root / "bddl_files"),
                "init_states": str(package_root / "init_files"), "assets": str(package_root / "assets")}
    if config_file.exists():
        previous = yaml.safe_load(config_file.read_text())
        if not isinstance(previous, dict) or any(
                key not in previous or Path(previous[key]).resolve() != Path(value).resolve()
                for key, value in expected.items()):
            raise ValueError("LIBERO path config belongs to a different checkout; use a separate LIBERO_CONFIG_PATH")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        from thinkwam_p1.evaluation_protocol import atomic_json
        # JSON is valid YAML; an atomic write avoids partial files between ranks.
        atomic_json(config_file, {**expected, "datasets": str(package_root.parent.parent / "datasets")})
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)



def load_config(path):
    import yaml
    values = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("jepa_horizons", "jepa_token_counts"):
        if key in values:
            values[key] = tuple(values[key])
    from thinkwam_p0.config import P0Config
    return P0Config(**values)


def training_contract(config):
    """Read the actual cache shapes and HDF5 camera convention, not guessed defaults."""
    import h5py

    # The current online recipe keeps the full two-view V-JEPA grid and has no
    # feature-cache manifest.  Validate its visual/text contract directly from
    # the model config, while still reading HDF5 metadata for camera semantics.
    if config.online_vjepa:
        expected_tokens = config.vjepa_views * (config.vjepa_image_size // config.vjepa_patch_size) ** 2
        if config.visual_token_count != expected_tokens:
            raise ValueError(
                f"online checkpoint visual_token_count={config.visual_token_count} "
                f"does not match V-JEPA grid {expected_tokens}"
            )
        visual_tokens = config.visual_token_count
        visual_dim = config.vjepa_dim
        text_length = config.text_max_length
        cache_manifest = ""
        feature_precision = "online V-JEPA BF16 encoder, evaluator roundtrip to float32"
    else:
        if not config.feature_cache:
            raise ValueError("Cache diagnostics require config.feature_cache")
        cache = Path(config.feature_cache)
        manifest = json.loads((cache / "manifest.json").read_text())
        first = manifest["records"][0]
        sharded = isinstance(first, dict)
        with np.load(cache / (first["file"] if sharded else first), allow_pickle=False) as data:
            def first_record(key):
                return data[key][0] if sharded else data[key]
            obs_shape = first_record("obs_features").shape
            text_length = len(first_record("text_input_ids"))
            for key, expected in (
                ("proprio", (config.proprio_dim,)),
                ("actions", (config.action_horizon, config.action_dim)),
                ("action_history", (config.memory_action_history, config.action_dim)),
            ):
                if first_record(key).shape != expected:
                    raise ValueError(f"Cache {key} shape disagrees with checkpoint: {first_record(key).shape} != {expected}")
            if obs_shape[1] != config.vjepa_dim:
                raise ValueError(f"Unexpected cached observation shape {obs_shape}")
        visual_tokens = obs_shape[0]
        visual_dim = obs_shape[1]
        cache_manifest = str(cache / "manifest.json")
        feature_precision = "float16 cache roundtrip to float32"
    hdf5_path = sorted(Path(config.dataset_root).rglob("*.hdf5"))[0]
    with h5py.File(hdf5_path, "r") as data:
        demo = data["data"][sorted(data["data"].keys())[0]]
        height, width = demo["obs/agentview_rgb"].shape[1:3]
        convention = str(data["data"].attrs["macros_image_convention"])
    if height != width:
        raise ValueError("This evaluator requires square LIBERO cameras")
    return {
        "visual_tokens": visual_tokens, "visual_dim": visual_dim,
        "text_length": text_length, "camera_resolution": height,
        "image_convention": convention, "rotate_180": False,
        "image_resize": "tensor bicubic, align_corners=False, antialias=False",
        "feature_precision": feature_precision,
        "proprio": "eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)",
        "gripper": "raw LIBERO sign, no inversion; sign binarization",
        "action_history": "last 8 executed commands; zero padded at episode start",
        "cache_manifest": cache_manifest,
    }


def video_frame(obs):
    # Flip OpenGL rows only for display; policy inputs keep the training convention.
    return np.concatenate([
        np.asarray(obs["agentview_image"])[::-1],
        np.asarray(obs["robot0_eye_in_hand_image"])[::-1],
    ], axis=1).astype(np.uint8)
