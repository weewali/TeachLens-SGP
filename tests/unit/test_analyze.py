"""CLI time parsing for --start/--end; loads the script without running it."""

import argparse
import importlib.util
import unittest

from src.utils.paths import project_path


spec = importlib.util.spec_from_file_location("analyze_cli", project_path("scripts/analyze.py"))
analyze_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_cli)


class TimeArgumentTests(unittest.TestCase):
    def test_accepts_seconds_and_clock_forms(self):
        for text, seconds in (("90", 90.0), ("90.5", 90.5), ("1:30", 90.0),
                              ("75:00", 4500.0), ("1:02:03.5", 3723.5)):
            self.assertEqual(analyze_cli.parse_seconds(text), seconds)

    def test_rejects_ambiguous_or_invalid_times(self):
        for text in ("", "-5", "1:75", "1:2:3:4", "abc", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                analyze_cli.parse_seconds(text)


if __name__ == "__main__":
    unittest.main()
