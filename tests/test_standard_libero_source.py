"""Small CPU HDF5 fixtures verify the standard-LIBERO data boundary."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np


# Load only the validator and its static allowlist, without torch/model imports.
_PACKAGE = "_libero_validation_tests"
_PACKAGE_PATH = Path(__file__).resolve().parents[1] / "thinkwam_p1"
package = ModuleType(_PACKAGE)
package.__path__ = [str(_PACKAGE_PATH)]
sys.modules[_PACKAGE] = package
_SPEC = importlib.util.spec_from_file_location(
    _PACKAGE + ".libero_protocol", _PACKAGE_PATH / "libero_protocol.py"
)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


class StandardLiberoSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "standard"
        for suite, tasks in protocol.STANDARD_TASKS.items():
            directory = self.root / suite
            directory.mkdir(parents=True)
            for task in tasks:
                with h5py.File(directory / (task + "_demo.hdf5"), "w") as handle:
                    data = handle.create_group("data")
                    data.attrs["num_demos"] = 50
                    data.attrs["bddl_file_name"] = f"libero/libero/bddl_files/{suite}/{task}.bddl"
                    data.attrs["problem_info"] = json.dumps({
                        "language_instruction": protocol.standard_task_language(task),
                        "domain_name": "robosuite",
                        "problem_name": "libero_tabletop_manipulation",
                    })
                    data.attrs["env_name"] = "Libero_Tabletop_Manipulation"
                    data.attrs["env_args"] = json.dumps({
                        "problem_name": "libero_tabletop_manipulation",
                        "env_name": "Libero_Tabletop_Manipulation",
                        "bddl_file": "original_collection_name.bddl",
                    })
                    for index in range(50):
                        data.create_group(f"demo_{index}")
        self.config = SimpleNamespace(dataset_root=str(self.root), online_vjepa=True)
        self.task_path = self.root / "libero_goal" / (
            protocol.STANDARD_TASKS["libero_goal"][0] + "_demo.hdf5"
        )

    def validate(self):
        return protocol.validate_standard_libero_source(self.config)

    def test_accepts_standard_tasks_sidecars_and_added_soft_affordance(self):
        (self.root / "manifest.json").write_text("{}")
        (self.root / "libero_goal/labels.json").write_text("{}")
        with h5py.File(self.task_path, "r+") as handle:
            handle["data/demo_0"].create_dataset(
                "affordance", data=np.full((2, 2, 24, 24), .3, dtype=np.float32)
            )
        self.assertEqual(self.validate(), dict.fromkeys(protocol.STANDARD_SUITES, 10))

    def test_allowlist_is_fixed_not_derived_from_local_names(self):
        self.assertEqual(len(protocol.STANDARD_TASKS), 4)
        self.assertEqual(sum(map(len, protocol.STANDARD_TASKS.values())), 40)
        self.assertIn("turn_on_the_stove", protocol.STANDARD_TASKS["libero_goal"])
        self.task_path.rename(self.task_path.with_name("unrelated_variant_demo.hdf5"))
        with self.assertRaisesRegex(ValueError, "exact 40 standard"):
            self.validate()

    def test_rejects_extra_or_nested_hdf5_inputs(self):
        for relative in ("extra.hdf5", "libero_goal/backup/extra.hdf5"):
            with self.subTest(relative=relative):
                extra = self.root / relative
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.touch()
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    self.validate()
                extra.unlink()

    def test_rejects_incomplete_task_coverage(self):
        self.task_path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate()

    def test_rejects_extra_demonstrations_inside_a_whitelisted_file(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data"].create_group("demo_50")
        with self.assertRaisesRegex(ValueError, "demo_0 through demo_49"):
            self.validate()

    def test_rejects_wrong_demo_count_metadata(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data"].attrs["num_demos"] = 51
        with self.assertRaisesRegex(ValueError, "num_demos=50"):
            self.validate()

    def test_rejects_relabelled_variant_bddl_metadata(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data"].attrs["bddl_file_name"] = "libero_plus/perturbed_variant.bddl"
        with self.assertRaisesRegex(ValueError, "BDDL task metadata"):
            self.validate()

    def test_rejects_wrong_language_or_missing_metadata(self):
        with h5py.File(self.task_path, "r+") as handle:
            info = json.loads(handle["data"].attrs["problem_info"])
            info["language_instruction"] = "a different task instruction"
            handle["data"].attrs["problem_info"] = json.dumps(info)
        with self.assertRaisesRegex(ValueError, "Task language"):
            self.validate()
        with h5py.File(self.task_path, "r+") as handle:
            del handle["data"].attrs["problem_info"]
        with self.assertRaisesRegex(ValueError, "problem_info"):
            self.validate()

    def test_rejects_inconsistent_environment_identity(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data"].attrs["env_name"] = "Another_Environment"
        with self.assertRaisesRegex(ValueError, "environment metadata"):
            self.validate()

    def test_rejects_external_filesystem_symlink(self):
        outside = self.parent / "external.hdf5"
        self.task_path.rename(outside)
        try:
            self.task_path.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Creating symlinks is unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.validate()

    def test_rejects_symlinked_suite_directory(self):
        original = self.root / "libero_goal"
        outside = self.parent / "other_suite"
        original.rename(outside)
        try:
            original.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Creating symlinks is unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.validate()

    def test_rejects_filesystem_hardlink_alias(self):
        outside = self.parent / "external_alias.hdf5"
        try:
            os.link(self.task_path, outside)
        except OSError as exc:
            self.skipTest(f"Creating hard links is unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "hard-linked"):
            self.validate()

    def test_rejects_hdf5_external_link_before_dereferencing_it(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data/demo_0"]["observations"] = h5py.ExternalLink(
                str(self.parent / "not_present.hdf5"), "/data"
            )
        with self.assertRaisesRegex(ValueError, "external/soft links"):
            self.validate()

    def test_rejects_hdf5_soft_link(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data/demo_0"]["observations"] = h5py.SoftLink("/data/demo_1")
        with self.assertRaisesRegex(ValueError, "external/soft links"):
            self.validate()

    def test_rejects_duplicate_demonstration_hard_links(self):
        with h5py.File(self.task_path, "r+") as handle:
            del handle["data/demo_49"]
            handle["data"]["demo_49"] = handle["data/demo_0"]
        with self.assertRaisesRegex(ValueError, "Aliased HDF5"):
            self.validate()

    def test_rejects_virtual_dataset_even_without_available_source(self):
        with h5py.File(self.task_path, "r+") as handle:
            source = h5py.VirtualSource(str(self.parent / "not_present.hdf5"), "images", shape=(1,))
            layout = h5py.VirtualLayout(shape=(1,), dtype=np.float32)
            layout[:] = source
            handle["data/demo_0"].create_virtual_dataset("images", layout)
        with self.assertRaisesRegex(ValueError, "virtual/external storage"):
            self.validate()

    def test_rejects_external_raw_storage(self):
        with h5py.File(self.task_path, "r+") as handle:
            handle["data/demo_0"].create_dataset(
                "images", shape=(1,), dtype=np.float32,
                external=[(str(self.parent / "outside.raw"), 0, h5py.h5f.UNLIMITED)],
            )
        with self.assertRaisesRegex(ValueError, "virtual/external storage"):
            self.validate()

    def test_rejects_feature_cache_without_raw_demonstrations(self):
        self.config.online_vjepa = False
        with self.assertRaisesRegex(ValueError, "online_vjepa"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
