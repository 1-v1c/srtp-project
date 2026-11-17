"""Compatibility entry point for integrations that import brake_algorithm."""
from aeb.brake_algorithm import (
    BrakeConfig,
    BrakeFrame,
    LongitudinalBrake,
    compute_brake,
    main,
    reset_brake_controller,
    update_brake_result,
)

__all__ = [
    "BrakeConfig",
    "BrakeFrame",
    "LongitudinalBrake",
    "compute_brake",
    "reset_brake_controller",
    "update_brake_result",
]


if __name__ == "__main__":
    main()
