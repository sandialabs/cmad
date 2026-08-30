"""``cmad --devices N``: the device count from the command line.

JAX reads the CPU device count only before its backend initialises, so the
option is exercised in a subprocess. In this process, where the backend is
already up, asking for more devices than JAX has is an error. Without the
option the device count is left to JAX.
"""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.fem import sharding
from cmad.io.exodus import read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType
from tests.cli.test_primal_fe_roundtrip import (
    _make_fe_primal_deck_coupled,
    _write_hex_cube_mesh,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs main() on the given command line, then reports the device count JAX
# ended up with.
_SUBPROCESS_SCRIPT = """
import sys
from cmad.cli.main import main
rc = main(sys.argv[1:])
import jax
print("JAX DEVICES", len(jax.devices()))
sys.exit(rc)
"""


def _write_deck(tmp: Path, name: str) -> Path:
    """The J2 + Voce cube of ``test_primal_fe_roundtrip`` on 2 x 2 x 2
    elements, writing ``u`` to ``tmp / name / primal.exo``."""
    mesh_path = tmp / "mesh.exo"
    if not mesh_path.exists():
        _write_hex_cube_mesh(mesh_path, divisions=(2, 2, 2))
    deck = _make_fe_primal_deck_coupled(
        mesh_filename=str(mesh_path),
        output_section={
            "path": str(tmp / name),
            "exodus filename": "primal.exo",
            "global residual": ["u"],
        },
    )
    deck_path = tmp / f"{name}.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return deck_path


def _read_u(out_dir: Path) -> np.ndarray:
    results = read_results(
        out_dir / "primal.exo",
        nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
    )
    return np.asarray(results.nodal["u"])


class TestDevicesOption(unittest.TestCase):

    def tearDown(self) -> None:
        sharding.set_device_count(None)

    def test_four_devices_subprocess_matches_one_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            deck_one = _write_deck(tmp, "one")
            deck_four = _write_deck(tmp, "four")
            self.assertEqual(cmad_main(["primal", str(deck_one)]), 0)
            proc = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_SCRIPT,
                 "--devices", "4", "primal", str(deck_four)],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}",
            )
            self.assertIn("JAX DEVICES 4", proc.stdout)
            u_one = _read_u(tmp / "one")
            u_four = _read_u(tmp / "four")
        self.assertEqual(u_one.shape, u_four.shape)
        scale = float(np.abs(u_one).max())
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(u_four, u_one, rtol=0, atol=1e-12 * scale)

    def test_in_process_count_above_the_devices_present_raises(self) -> None:
        jax.devices()  # the backend is up in this process
        with self.assertRaisesRegex(RuntimeError, "--devices 4"):
            cmad_main(["--devices", "4", "primal", "input.yaml"])
        self.assertIsNone(sharding._device_count)

    def test_without_the_option_the_count_is_left_to_jax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            deck = _write_deck(tmp, "default")
            self.assertEqual(cmad_main(["primal", str(deck)]), 0)
            self.assertIsNone(sharding._device_count)
            self.assertEqual(cmad_main(["--devices", "1", "primal", str(deck)]), 0)
            self.assertEqual(sharding._device_count, 1)

    def test_rejects_a_count_below_one(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as raised:
            cmad_main(["--devices", "0", "primal", "input.yaml"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
