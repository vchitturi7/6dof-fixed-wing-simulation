# test_dynamics_6dof.py
# Basic sanity tests for the 6DOF rigid-body dynamics.
#
# Usage from project root:
#   python src/sixdof/test_dynamics_6dof.py
#
# These tests verify:
#   - state derivative has correct shape
#   - zero attitude position kinematics are intuitive
#   - gravity points down in the body frame for level attitude
#   - throttle increases forward acceleration
#   - aileron creates roll acceleration

import numpy as np

from dynamics_6dof import (
    dynamics_6dof,
    gravity_body,
    make_initial_state,
)
import aircraft_6dof_params as p


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


def base_controls():
    """
    Neutral control dictionary.
    """

    return {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": 0.5,
        "wind_inertial": np.zeros(3),
    }


def test_state_derivative_shape():
    """
    dynamics_6dof should return a 12-element derivative vector.
    """

    state = make_initial_state()
    controls = base_controls()

    state_dot = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls,
    )

    if state_dot.shape != (12,):
        raise AssertionError(
            f"Expected state_dot shape (12,), got {state_dot.shape}"
        )


def test_position_kinematics_zero_attitude():
    """
    With zero attitude and body velocity [25, 0, 0], inertial position
    derivative should be [25, 0, 0].
    """

    state = make_initial_state(
        airspeed_mps=25.0,
        altitude_m=100.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
    )

    controls = base_controls()

    state_dot = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls,
    )

    position_dot = state_dot[0:3]

    assert_close(
        actual=position_dot,
        expected=np.array([25.0, 0.0, 0.0]),
        tolerance=1e-12,
        message="Position kinematics failed for zero attitude.",
    )


def test_gravity_level_attitude():
    """
    For zero roll, pitch, and yaw, gravity in body axes should be [0, 0, g].
    """

    g_body = gravity_body(
        phi=0.0,
        theta=0.0,
        psi=0.0,
    )

    assert_close(
        actual=g_body,
        expected=np.array([0.0, 0.0, p.G]),
        tolerance=1e-12,
        message="Gravity body vector failed for level attitude.",
    )


def test_throttle_increases_forward_acceleration():
    """
    Higher throttle should increase u_dot.
    """

    state = make_initial_state(
        airspeed_mps=25.0,
        altitude_m=100.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
    )

    controls_low = base_controls()
    controls_high = base_controls()

    controls_low["delta_t"] = 0.2
    controls_high["delta_t"] = 0.8

    state_dot_low = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls_low,
    )

    state_dot_high = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls_high,
    )

    u_dot_low = state_dot_low[3]
    u_dot_high = state_dot_high[3]

    if u_dot_high <= u_dot_low:
        raise AssertionError(
            "Higher throttle should increase forward acceleration u_dot."
        )


def test_aileron_creates_roll_acceleration():
    """
    Positive aileron should change p_dot relative to neutral aileron.
    """

    state = make_initial_state(
        airspeed_mps=25.0,
        altitude_m=100.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
    )

    controls_neutral = base_controls()
    controls_aileron = base_controls()

    controls_neutral["delta_a"] = 0.0
    controls_aileron["delta_a"] = np.deg2rad(10.0)

    state_dot_neutral = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls_neutral,
    )

    state_dot_aileron = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls_aileron,
    )

    p_dot_neutral = state_dot_neutral[9]
    p_dot_aileron = state_dot_aileron[9]

    if abs(p_dot_aileron) <= abs(p_dot_neutral):
        raise AssertionError(
            "Aileron command should increase roll acceleration magnitude."
        )


def main():
    print("Running 6DOF dynamics tests...")
    print("------------------------------")

    test_state_derivative_shape()
    print("Passed: state derivative shape")

    test_position_kinematics_zero_attitude()
    print("Passed: position kinematics zero attitude")

    test_gravity_level_attitude()
    print("Passed: gravity level attitude")

    test_throttle_increases_forward_acceleration()
    print("Passed: throttle increases forward acceleration")

    test_aileron_creates_roll_acceleration()
    print("Passed: aileron creates roll acceleration")

    print("\nAll 6DOF dynamics tests passed.")


if __name__ == "__main__":
    main()