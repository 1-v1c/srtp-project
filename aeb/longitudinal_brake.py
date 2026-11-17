"""Stateful braking profiles for confirmed longitudinal AEB targets."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BrakeTuning:
    light_decel: float = 3.0
    low_speed_decel: float = 1.5
    service_decel: float = 5.5
    light_jerk: float = 6.0
    service_jerk: float = 15.0
    release_jerk: float = 5.0
    standstill_gap: float = 1.0
    min_predicted_gap: float = 0.35
    release_margin: float = 0.75
    release_confirm: float = 0.30
    release_preview: float = 1.0
    lead_time: float = 0.15
    fatigue_lead_per_level: float = 0.12
    warning_lead: float = 0.35
    preview_step: float = 0.05


@dataclass(frozen=True)
class BrakeFrame:
    gap: float
    ego_speed: float
    lead_speed: float
    ego_accel: float
    lead_accel: float
    fatigue_level: int
    actuator_delay: float
    available_decel: float
    actuator_jerk: float


@dataclass(frozen=True)
class BrakeCommand:
    level: int
    phase: str
    warning: bool
    reference: float
    decel: float
    light_gap_margin: float


def _approach(current, target, rise, fall):
    return current + min(rise, max(-fall, target - current))


class LongitudinalBrake:
    def __init__(self, tuning=None):
        self.tuning = tuning or BrakeTuning()
        self.level = 0
        self.decel = 0.0
        self.clear_time = 0.0

    def _minimum_gap(self, frame, target_decel, build_jerk, horizon=None):
        """Roll out a piecewise rate-limited ego and a braking lead until clear or stopped."""
        tuning = self.tuning
        release_jerk = min(tuning.release_jerk, frame.actuator_jerk)
        ego, lead, accel = frame.ego_speed, frame.lead_speed, frame.ego_accel
        gap = minimum_gap = frame.gap
        lead_accel = min(frame.lead_accel, 0.0)
        t = 0.0
        while horizon is None or t < horizon - 1e-9:
            if ego <= 0.0 and accel <= 0.0:
                break
            if lead_accel == 0.0 and ego <= lead and accel <= 0.0:
                break

            step = tuning.preview_step
            if horizon is not None:
                step = min(step, horizon - t)
            # Place the end of actuator delay exactly on a prediction boundary.
            if t < frame.actuator_delay - 1e-9:
                step = min(step, frame.actuator_delay - t)
                next_accel = accel
            else:
                next_accel = _approach(accel, -target_decel, release_jerk * step, build_jerk * step)

            next_ego = max(0.0, ego + (accel + next_accel) * step / 2)
            next_lead = max(0.0, lead + lead_accel * step)
            closing = ego - lead
            next_closing = next_ego - next_lead
            if closing > 0 > next_closing:
                zero_time = step * closing / (closing - next_closing)
                minimum_gap = min(minimum_gap, gap - closing * zero_time / 2)
            gap -= (closing + next_closing) * step / 2
            minimum_gap = min(minimum_gap, gap)
            if minimum_gap <= 0.0:
                break
            ego, lead, accel = next_ego, next_lead, next_accel
            t += step
        return minimum_gap

    def step(self, frame, dt_sec):
        if not math.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("dt_sec must be positive and finite")
        tuning = self.tuning
        closing = frame.ego_speed - frame.lead_speed
        light_decel = tuning.low_speed_decel if closing <= 1.0 else tuning.light_decel
        targets = (0.0, min(light_decel, frame.available_decel),
                   min(tuning.service_decel, frame.available_decel), frame.available_decel)
        build_jerks = (frame.actuator_jerk, min(tuning.light_jerk, frame.actuator_jerk),
                       min(tuning.service_jerk, frame.actuator_jerk), frame.actuator_jerk)

        light_gap = self._minimum_gap(frame, targets[1], build_jerks[1])
        lead_time = tuning.lead_time + frame.fatigue_level * tuning.fatigue_lead_per_level
        approaching = closing > 0 or light_gap < frame.gap - 1e-6
        closing_for_preview = max(0.0, closing, -frame.lead_accel * frame.actuator_delay)
        enter_gap = tuning.standstill_gap + lead_time * closing_for_preview
        warning_gap = enter_gap + tuning.warning_lead * closing_for_preview
        demand = 0
        if approaching and light_gap <= enter_gap:
            demand = 1
            if light_gap < tuning.min_predicted_gap:
                service_gap = self._minimum_gap(frame, targets[2], build_jerks[2])
                demand = 2 if service_gap >= tuning.min_predicted_gap else 3
        elif self.level and frame.ego_speed > 0:
            coast_gap = self._minimum_gap(frame, 0.0, frame.actuator_jerk, tuning.release_preview)
            release_floor = min(frame.gap, tuning.standstill_gap)
            coast_gap_loss = frame.gap - coast_gap
            release_safe = coast_gap >= release_floor - 1e-6 and (
                not approaching or light_gap - coast_gap_loss >= enter_gap + tuning.release_margin
            )
            if not release_safe:
                demand = 1

        if demand >= self.level:
            self.clear_time = 0.0
            self.level = demand
        else:
            self.clear_time += dt_sec
            if self.clear_time + 1e-9 >= tuning.release_confirm:
                self.level = demand
                self.clear_time = 0.0

        reference = targets[self.level]
        self.decel = min(frame.available_decel, _approach(
            self.decel, reference, build_jerks[self.level] * dt_sec,
            min(tuning.release_jerk, frame.actuator_jerk) * dt_sec,
        ))
        warning = bool(self.level) or (approaching and light_gap <= warning_gap)
        phase = ("idle", "light", "service", "emergency")[self.level]
        if self.level == 0:
            phase = "release" if self.decel > 0 else ("warning" if warning else phase)
        return BrakeCommand(self.level, phase, warning, reference, self.decel,
                            light_gap - tuning.standstill_gap)
