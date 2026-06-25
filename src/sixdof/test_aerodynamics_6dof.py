# test_aerodynamics_6dof.py
# Basic sanity tests for the 6DOF aerodynamic model.
#
# Usage from project root:
#   python src/sixdof/test_aerodynamics_6dof.py
#
# These tests are not meant to prove the aero model is perfect.
# They check that:
#   - air data calculation is sensible
#   - force/moment outputs have correct shape
#   - increasing throttle increases forward force
#   - positive angle of attack creates upward lift, which is negative Z force
#   - aileron creates roll moment

import numpy as np

from aerodynamics_6dof import (
    compute_air_data,
    aerodynamic_forces_moments,
    total_forces_moments,
)


def assert_close(actual, expected, tolerance, message):
    """
    Small helper for readable test failures.
    """

    if not np.allclose(actual, expected, atol=tolerance):
        raise AssertionError(
            f"{message}\n"
            f"Actual:   {actual}\n"
            f"Expected: {expected}\n"
            f"Tolerance: {tolerance}"
        )


def test_air_data_zero_wind():
    """
    With zero wind and velocity [25, 0, 0], airspeed should be 25 m/s,
    alpha should be 0, and beta should be 0.
    """

    velocity_body = np.array([25.0, 0.0, 0.0])
    wind_body = np.array([0.0, 0.0, 0.0])

    air_data = compute_air_data(
        velocity_body=velocity_body,
        wind_body=wind_body,
    )

    assert_close(
        actual=air_data["V"],
        expected=25.0,
        tolerance=1e-12,
        message="Airspeed calculation failed.",
    )

    assert_close(
        actual=air_data["alpha"],
        expected=0.0,
        tolerance=1e-12,
        message="Alpha should be zero for [u, v, w] = [25, 0, 0].",
    )

    assert_close(
        actual=air_data["beta"],
        expected=0.0,
        tolerance=1e-12,
        message="Beta should be zero for [u, v, w] = [25, 0, 0].",
    )


def test_output_shapes():
    """
    Forces and moments should both be 3-element vectors.
    """

    velocity_body = np.array([25.0, 0.0, 1.0])
    angular_rates_body = np.array([0.0, 0.0, 0.0])
    controls = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
    }

    output = aerodynamic_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls,
        wind_body=np.zeros(3),
    )

    if output["forces_body"].shape != (3,):
        raise AssertionError("forces_body should have shape (3,)")

    if output["moments_body"].shape != (3,):
        raise AssertionError("moments_body should have shape (3,)")


def test_positive_alpha_gives_upward_lift():
    """
    In body axes, +Z is downward.

    Positive alpha should generally create positive lift, which means the
    aerodynamic Z force should be negative/upward.
    """

    velocity_body = np.array([25.0, 0.0, 1.0])
    angular_rates_body = np.array([0.0, 0.0, 0.0])
    controls = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
    }

    output = aerodynamic_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls,
        wind_body=np.zeros(3),
    )

    z_force = output["forces_body"][2]

    if z_force >= 0.0:
        raise AssertionError(
            "Expected positive-alpha lift to create negative/upward Z force."
        )


def test_throttle_increases_forward_force():
    """
    Increasing throttle should increase total X force.
    """

    velocity_body = np.array([25.0, 0.0, 0.0])
    angular_rates_body = np.array([0.0, 0.0, 0.0])

    controls_low = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": 0.2,
    }

    controls_high = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": 0.8,
    }

    output_low = total_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls_low,
        wind_body=np.zeros(3),
    )

    output_high = total_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls_high,
        wind_body=np.zeros(3),
    )

    x_low = output_low["forces_body"][0]
    x_high = output_high["forces_body"][0]

    if x_high <= x_low:
        raise AssertionError("Higher throttle should increase forward X force.")


def test_aileron_creates_roll_moment():
    """
    A nonzero aileron command should create a roll moment.
    """

    velocity_body = np.array([25.0, 0.0, 0.0])
    angular_rates_body = np.array([0.0, 0.0, 0.0])

    controls_neutral = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
    }

    controls_aileron = {
        "delta_e": 0.0,
        "delta_a": np.deg2rad(10.0),
        "delta_r": 0.0,
    }

    output_neutral = aerodynamic_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls_neutral,
        wind_body=np.zeros(3),
    )

    output_aileron = aerodynamic_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls_aileron,
        wind_body=np.zeros(3),
    )

    roll_neutral = output_neutral["moments_body"][0]
    roll_aileron = output_aileron["moments_body"][0]

    if abs(roll_aileron) <= abs(roll_neutral):
        raise AssertionError("Aileron command should increase roll moment magnitude.")


def main():
    print("Running 6DOF aerodynamic model tests...")
    print("---------------------------------------")

    test_air_data_zero_wind()
    print("Passed: air data zero wind")

    test_output_shapes()
    print("Passed: force/moment output shapes")

    test_positive_alpha_gives_upward_lift()
    print("Passed: positive alpha gives upward lift")

    test_throttle_increases_forward_force()
    print("Passed: throttle increases forward force")

    test_aileron_creates_roll_moment()
    print("Passed: aileron creates roll moment")

    print("\nAll 6DOF aerodynamic tests passed.")


if __name__ == "__main__":
    main()