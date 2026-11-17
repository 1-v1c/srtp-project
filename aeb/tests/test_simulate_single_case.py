import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aeb.simulate_single_case import main, parse_args
from aeb.simulate_staged_brake import Scenario


class SingleCaseTests(unittest.TestCase):
    def test_configurable_lead_braking_uses_requested_interval(self):
        case = Scenario("test", "test", 22.0, 60 / 3.6, 40 / 3.6, lead_brake=4.0,
                        lead_brake_start_s=0.5, lead_brake_end_s=2.0)
        self.assertEqual([case.lead_accel(t) for t in (0.49, 0.5, 1.99, 2.0)],
                         [0.0, -4.0, -4.0, 0.0])

    @patch("aeb.simulate_single_case.render_plot")
    def test_single_case_exports_one_summary_and_aligned_trace(self, plot):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            main(["--output-dir", folder, "--controller", "v2", "--fatigue", "0"])
            for name in ("aeb-single-summary.csv", "aeb-single-traces.csv"):
                with (output / name).open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if name == "aeb-single-summary.csv":
                    self.assertEqual(len(rows), 1)
                    result = rows[0]
                else:
                    trace = rows
            self.assertEqual(result["controller"], "v2")
            self.assertEqual(result["outcome"], "contact")
            self.assertAlmostEqual(float(trace[0]["ego_speed_kmh"]), 60)
            self.assertEqual(float(trace[0]["bumper_gap_m"]), 22)
            self.assertLess(min(float(row["command_accel_mps2"]) for row in trace), 0)
            self.assertEqual(trace[-1]["contact"], "True")
            self.assertEqual(trace[-1]["bumper_gap_m"], "0.0")
            self.assertEqual(trace[-1]["time_s"], result["end_time_s"])
            self.assertEqual(plot.call_args.args[0], output / "aeb-single-curves.png")
            plot.assert_called_once()

    def test_nonfinite_input_is_rejected(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            parse_args(["--output-dir", "unused", "--delay-s", "nan"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
