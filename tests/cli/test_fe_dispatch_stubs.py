"""``cmad calibrate`` rejects FE decks early.

The remaining FE-unsupported subcommand raises
``NotImplementedError``. The test checks the exception type and
that the message names the subcommand and contains the ``'fe'``
literal.
"""
import tempfile
import unittest
from pathlib import Path

import yaml

from cmad.cli.main import main as cmad_main


class TestFeDispatchStubs(unittest.TestCase):
    def _check(self, subcommand: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deck_path = Path(tmpdir) / "deck.yaml"
            deck_path.write_text(
                yaml.safe_dump({"problem": {"type": "fe"}}),
            )
            with self.assertRaises(NotImplementedError) as ctx:
                cmad_main([subcommand, str(deck_path)])
            msg = str(ctx.exception)
            self.assertIn(f"cmad {subcommand}", msg)
            self.assertIn("'fe'", msg)
            self.assertIn("not yet supported", msg)

    def test_calibrate_rejects_fe(self) -> None:
        self._check("calibrate")


if __name__ == "__main__":
    unittest.main()
