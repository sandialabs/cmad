"""``cmad --cpu-devices N``: JAX's CPU device count from the command line.

JAX reads the count only before its backend initialises, so the option is
exercised in a subprocess; in this process, where the backend is already
up, it warns and the run proceeds on the devices present.
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


class TestCpuDevicesOption(unittest.TestCase):

    def test_four_devices_subprocess_matches_one_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            deck_one = _write_deck(tmp, "one")
            deck_four = _write_deck(tmp, "four")
            self.assertEqual(cmad_main(["primal", str(deck_one)]), 0)
            proc = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_SCRIPT,
                 "--cpu-devices", "4", "primal", str(deck_four)],
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

    def test_in_process_warns_and_runs_on_the_devices_present(self) -> None:
        jax.devices()  # the backend is up in this process
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            deck = _write_deck(tmp, "warned")
            with self.assertWarnsRegex(UserWarning, "--cpu-devices 4 ignored"):
                rc = cmad_main(["--cpu-devices", "4", "primal", str(deck)])
            self.assertEqual(rc, 0)
            self.assertTrue((tmp / "warned" / "primal.exo").exists())

    def test_rejects_a_count_below_one(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as raised:
            cmad_main(["--cpu-devices", "0", "primal", "input.yaml"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
