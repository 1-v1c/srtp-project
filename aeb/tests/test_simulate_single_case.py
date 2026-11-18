import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aeb.simulate_single_case import main, parse_args, trace_rows
from aeb.simulate_staged_brake import Scenario, simulate


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

    def test_driver_request_is_sampled_and_reaches_vehicle(self):
        case = Scenario("manual", "manual", 80.0, 10.0, 10.0, target_valid=False,
                        driver_brake_mps2=3.0, driver_brake_start_s=0.305,
                        driver_brake_end_s=0.805)
        for staged in (False, True):
            with self.subTest(staged=staged):
                records = simulate(case, staged=staged, fatigue=0)
                trace = list(trace_rows(records, case))
                first_request = next(row for row in trace if row["driver_request_mps2"] > 0)
                first_command = next(row for row in records if row[4] > 0)
                self.assertAlmostEqual(first_request["time_s"], 0.32)
                self.assertAlmostEqual(first_command[0], 0.32)
                self.assertEqual(first_command[4], 3.0)
                self.assertTrue(all(row["driver_request_mps2"] == 0
                                    for row in trace if row["time_s"] >= 0.82))
                self.assertLess(records[-1][2], 10.0)
                self.assertLess(min(row[5] for row in records), -0.3)

    def test_stronger_driver_brake_overrides_aeb_in_v3(self):
        case = Scenario("driver", "driver", 45.0, 50/3.6, 0.0,
                        driver_brake_mps2=7.0, driver_brake_end_s=1.0)
        records = simulate(case, staged=True, fatigue=0)
        self.assertEqual(records[0][4], 7.0)
        self.assertGreater(-min(row[5] for row in records), 3.0)

    def test_invalid_driver_parameters_are_rejected(self):
        for parameters in (
            ["--driver-brake-mps2", "-1"],
            ["--driver-brake-mps2", "nan"],
            ["--driver-brake-start-s", "4", "--driver-brake-end-s", "3"],
            ["--driver-brake-end-s", "31"],
        ):
            with self.subTest(parameters=parameters), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    parse_args(["--output-dir", "unused", *parameters])
                self.assertEqual(error.exception.code, 2)

    def test_acceleration_reaches_vehicle_and_trace(self):
        case = Scenario("accel", "accel", 300.0, 10.0, 20.0, target_valid=False,
                        driver_accel_mps2=1.0, driver_accel_start_s=0.305,
                        driver_accel_end_s=1.005)
        for staged in (False, True):
            with self.subTest(staged=staged):
                records = simulate(case, staged=staged, fatigue=0)
                trace = list(trace_rows(records, case))
                first = next(r for r in trace if r["driver_longitudinal_request_mps2"] > 0)
                self.assertAlmostEqual(first["time_s"], 0.32)
                self.assertEqual(first["command_accel_mps2"], 1.0)
                self.assertGreater(records[-1][2], 10.5)
                self.assertGreater(max(r[5] for r in records), 0.9)
                self.assertTrue(all(r["driver_accel_request_mps2"] == 0
                                    for r in trace if r["time_s"] >= 1.02))

    def test_aeb_overrides_acceleration(self):
        for staged in (False, True):
            case = Scenario("conflict", "conflict", 5.0, 15.0, 0.0,
                            driver_accel_mps2=2.0)
            records = simulate(case, staged=staged, fatigue=0)
            self.assertGreater(records[0][4], 0)
            self.assertLess(min(row[5] for row in records), 0)

    def test_braking_wins_when_driver_intervals_overlap(self):
        case = Scenario("both", "both", 300.0, 10.0, 20.0, target_valid=False,
                        driver_accel_mps2=1.0, driver_accel_end_s=1.0,
                        driver_brake_mps2=2.0, driver_brake_start_s=0.5,
                        driver_brake_end_s=0.8)
        self.assertEqual(case.driver_operation(0.4), -1.0)
        self.assertEqual(case.driver_operation(0.5), 2.0)
        self.assertEqual(case.driver_operation(0.8), -1.0)
        records = simulate(case, staged=True, fatigue=0)
        trace = list(trace_rows(records, case))
        at_overlap = next(r for r in trace if r["time_s"] == 0.6)
        self.assertEqual(at_overlap["driver_longitudinal_request_mps2"], -2.0)
        self.assertEqual(at_overlap["command_accel_mps2"], -2.0)

    def test_invalid_acceleration_parameters_are_rejected(self):
        for parameters in (
            ["--driver-accel-mps2", "-1"],
            ["--driver-accel-mps2", "inf"],
            ["--driver-accel-start-s", "5", "--driver-accel-end-s", "4"],
            ["--driver-accel-end-s", "31"],
        ):
            with self.subTest(parameters=parameters), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args(["--output-dir", "unused", *parameters])

    def test_system_recommendation_excludes_driver_operation(self):
        for staged in (False, True):
            for operation in ({"driver_accel_mps2": 1.0}, {"driver_brake_mps2": 2.0}):
                with self.subTest(staged=staged, operation=operation):
                    case = Scenario("manual", "manual", 300.0, 10.0, 20.0,
                                    target_valid=False, **operation)
                    rows = list(trace_rows(simulate(case, staged, fatigue=0), case))
                    self.assertTrue(all(r["aeb_accel_mps2"] == 0 for r in rows))
                    self.assertNotEqual(rows[0]["command_accel_mps2"], 0)
                    self.assertTrue(any(abs(r["actual_accel_mps2"]) > 0.5 for r in rows))

    def test_system_recommendation_retains_jerk_shaping(self):
        case = Scenario("stop", "stop", 45.0, 50 / 3.6, 0.0,
                        driver_brake_mps2=7.0, driver_brake_end_s=1.0)
        rows = list(trace_rows(simulate(case, staged=True, fatigue=0), case))
        self.assertEqual(rows[0]["command_accel_mps2"], -7.0)
        self.assertTrue(any(r["reference_accel_mps2"] < r["aeb_accel_mps2"] < 0
                            for r in rows))
        self.assertTrue(any(r["command_accel_mps2"] < r["aeb_accel_mps2"]
                            for r in rows if r["time_s"] < 1.0))

    def test_nonfinite_input_is_rejected(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            parse_args(["--output-dir", "unused", "--delay-s", "nan"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
