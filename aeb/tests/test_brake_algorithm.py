import unittest

from aeb.brake_algorithm import compute_brake, update_brake_result
from aeb.data_box import DataBox


def make_box(*, distance=32.0, v_ego=20.0, v_target=10.0, fatigue=0):
    box = DataBox()
    box.target_valid = True
    box.d = box.dx = distance
    box.v_ego = v_ego
    box.v_target = v_target
    box.fatigue_level = fatigue
    return box


def simulate_following(
    update, *, fatigue, distance=40.0, v_ego=20.0, v_target=10.0,
    target_acceleration=lambda _: 0.0,
):
    """Run a 30 s, 50 Hz longitudinal simulation with a bumper-to-bumper gap."""
    box = make_box(distance=distance, v_ego=v_ego, v_target=v_target, fatigue=fatigue)
    dt = 0.02
    first_brake = None
    min_gap = distance
    for frame in range(1500):
        now = frame * dt
        box.a_target = target_acceleration(now) if box.v_target > 0.0 else 0.0
        update(box)
        if first_brake is None and box.brake_level:
            first_brake = (now, box.d, box.ttc)

        next_ego = max(0.0, box.v_ego - box.a_final * dt)
        next_target = max(0.0, box.v_target + box.a_target * dt)
        box.d += ((box.v_target + next_target) - (box.v_ego + next_ego)) * 0.5 * dt
        box.dx = box.d
        box.v_ego = next_ego
        box.v_target = next_target
        min_gap = min(min_gap, box.d)
        if box.d <= 0.0:
            break

    return first_brake, min_gap, box.v_ego - box.v_target


class BrakeAlgorithmTests(unittest.TestCase):
    def test_wrapper_uses_both_relative_position_components(self):
        box = make_box(distance=5.0, v_ego=10.0, v_target=0.0)
        box.dx, box.dy = 3.0, 4.0
        update_brake_result(box)
        self.assertAlmostEqual(box.ttc, 0.833)
        self.assertTrue(box.warning)

    def test_accelerator_without_closing_does_not_become_brake(self):
        box = make_box(v_target=20.0)
        box.a_driver = -2.0
        update_brake_result(box)
        self.assertFalse(box.warning)
        self.assertEqual(box.a_final, 0.0)

    def test_accelerator_preserves_requested_aeb_brake(self):
        reference = make_box(distance=20.0, fatigue=3)
        update_brake_result(reference)
        expected = reference.a_final
        box = make_box(distance=20.0, fatigue=3)
        box.a_driver = -7.0
        update_brake_result(box)
        self.assertTrue(box.warning)
        self.assertGreater(expected, 0.0)
        self.assertEqual(box.a_final, expected)

    def test_target_loss_clears_stale_aeb_output_and_preserves_manual_brake(self):
        box = make_box(distance=8.0, v_target=0.0)
        update_brake_result(box)
        self.assertGreater(box.brake_level, 0)
        box.target_valid = False
        box.a_driver = 2.0
        update_brake_result(box)
        self.assertEqual((box.a_final, box.warning, box.brake_level, box.ttc),
                         (2.0, False, 0, 999.0))

    def test_out_of_path_target_filtered_upstream_does_not_brake(self):
        box = make_box(distance=13.0, v_target=0.0)
        box.dx, box.dy = 12.0, 5.0
        box.target_valid = False
        update_brake_result(box)
        self.assertEqual((box.warning, box.brake_level, box.a_final), (False, 0, 0.0))

    def test_distance_jitter_around_30m_no_longer_switches_brake(self):
        for distance in (30.01, 29.99, 30.03, 29.97, 30.0):
            with self.subTest(distance=distance):
                box = make_box(distance=distance)
                update_brake_result(box)
                self.assertEqual((box.brake_level, box.a_final), (0, 0.0))

    def test_relative_speed_threshold_is_continuous(self):
        def warning_time(closing_speed):
            return compute_brake(0, 10.0, 10.0 - closing_speed, 0.0,
                                 20.0, 20.0, 0.0, 0.0, 0.0, 0.0)["t_warning"]

        self.assertLess(abs(warning_time(3.001) - warning_time(2.999)), 0.005)

    def test_invalid_fatigue_level_is_rejected(self):
        for fatigue in (-1, 4, 1.0, True):
            with self.subTest(fatigue=fatigue), self.assertRaises(ValueError):
                update_brake_result(make_box(fatigue=fatigue))

    def test_low_closing_speed_brakes_on_stopping_distance(self):
        for speed, gap, expected in (
            (0.075, 0.8, 1.5), (0.1, 0.8, 1.5), (0.101, 0.8, 1.5),
            (0.4, 1.15, 1.5), (0.4, 1.25, 0.0),
        ):
            with self.subTest(speed=speed, gap=gap):
                box = make_box(distance=gap, v_ego=speed, v_target=0.0)
                update_brake_result(box, staged=False)
                self.assertEqual(box.a_final, expected)
                self.assertEqual(box.warning, expected > 0)
                self.assertEqual(box.brake_level, int(expected > 0))

    def test_low_speed_uses_net_gap_independently_of_center_distance(self):
        box = make_box(distance=5.3, v_ego=10.075, v_target=10.0)
        box.bumper_gap_m = 0.8
        update_brake_result(box, staged=False)
        self.assertEqual((box.a_final, box.ttc), (1.5, 999.0))
        box.target_valid = False
        update_brake_result(box, staged=False)
        self.assertEqual(box.a_final, 0.0)

    def test_low_speed_ignores_non_closing_or_behind_target(self):
        for ego, lead, dx in ((0.0, 0.0, 0.5), (10.0, 10.0, 0.5),
                              (10.0, 10.075, 0.5), (10.075, 10.0, -0.5)):
            with self.subTest(ego=ego, lead=lead, dx=dx):
                box = make_box(distance=abs(dx), v_ego=ego, v_target=lead)
                box.dx = dx
                update_brake_result(box)
                self.assertEqual((box.warning, box.brake_level, box.a_final), (False, 0, 0.0))

    def test_low_speed_retains_stronger_ttc_brake(self):
        box = make_box(distance=0.25, v_ego=0.9, v_target=0.0)
        update_brake_result(box, staged=False)
        self.assertEqual((box.brake_level, box.a_final), (3, 8.0))

    def test_staged_low_speed_uses_bumper_gap_and_ramps(self):
        box = make_box(distance=5.3, v_ego=10.075, v_target=10.0)
        box.bumper_gap_m = 0.8
        update_brake_result(box)
        self.assertEqual((box.brake_phase, box.brake_reference, box.ttc), ("light", 1.5, 999.0))
        self.assertAlmostEqual(box.a_final, 0.12)
        update_brake_result(box)
        self.assertAlmostEqual(box.a_final, 0.24)

    def test_staged_hold_then_release_with_bounded_slope(self):
        box = make_box(distance=5.3, v_ego=10.075, v_target=10.0)
        box.bumper_gap_m = 0.8
        for _ in range(6):
            update_brake_result(box)
        box.v_ego = 9.0
        for _ in range(14):
            update_brake_result(box)
            self.assertEqual(box.brake_phase, "light")
        update_brake_result(box)
        self.assertEqual(box.brake_phase, "release")
        self.assertAlmostEqual(box.aeb_decel, 1.5 - 5*0.02)

    def test_staged_emergency_skips_light_stage_and_respects_driver_brake(self):
        box = make_box(distance=0.25, v_ego=0.9, v_target=0.0)
        box.a_driver = 3.5
        update_brake_result(box)
        self.assertEqual((box.brake_phase, box.brake_reference), ("emergency", 8.0))
        self.assertAlmostEqual(box.aeb_decel, 0.60)
        self.assertEqual(box.a_final, 3.5)

    def test_staged_stale_fatigue_uses_level_zero(self):
        fresh = make_box(distance=45.0, v_ego=50/3.6, v_target=0, fatigue=3)
        stale = make_box(distance=45.0, v_ego=50/3.6, v_target=0, fatigue=3)
        stale.fatigue_valid = False
        update_brake_result(fresh)
        update_brake_result(stale)
        self.assertEqual(fresh.brake_phase, "light")
        self.assertEqual(stale.brake_level, 0)

    def test_staged_target_change_discards_previous_latch(self):
        box = make_box(distance=5.3, v_ego=10.075, v_target=10.0)
        box.bumper_gap_m = 0.8
        box.target_id = 1
        update_brake_result(box)
        self.assertGreater(box.a_final, 0.0)
        box.target_id = 2
        box.v_ego = 10.0
        update_brake_result(box)
        self.assertEqual((box.a_final, box.brake_phase), (0.0, "idle"))

    def test_closed_loop_following_fatigue_brakes_earlier(self):
        results = [
            simulate_following(update_brake_result, fatigue=level)
            for level in range(4)
        ]
        for first_brake, min_gap, closing_at_end in results:
            self.assertIsNotNone(first_brake)
            self.assertGreater(min_gap, 0.0)
            self.assertLessEqual(closing_at_end, 1e-9)
        brake_times = [first_brake[0] for first_brake, _, _ in results]
        self.assertEqual(brake_times, sorted(brake_times, reverse=True))
        self.assertLess(brake_times[3], brake_times[0])

    def test_closed_loop_stationary_and_braking_lead_vehicle(self):
        scenarios = {
            "stationary": dict(distance=45.0, v_ego=50 / 3.6, v_target=0.0),
            "lead braking": dict(
                distance=36.0, v_ego=22.0, v_target=16.0,
                target_acceleration=lambda t: -5.0 if 1.2 <= t < 3.7 else 0.0,
            ),
        }
        for name, inputs in scenarios.items():
            for fatigue in range(4):
                with self.subTest(scenario=name, fatigue=fatigue):
                    first_brake, min_gap, closing_at_end = simulate_following(
                        update_brake_result, fatigue=fatigue, **inputs,
                    )
                    self.assertIsNotNone(first_brake)
                    self.assertGreater(min_gap, 0.0)
                    self.assertLessEqual(closing_at_end, 1e-9)


if __name__ == "__main__":
    unittest.main()
