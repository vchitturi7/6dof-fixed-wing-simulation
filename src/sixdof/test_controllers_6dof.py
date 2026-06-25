# test_controllers_6dof.py
# Basic tests for 6DOF controller functions.
#
# Usage from project root:
#   python src/sixdof/test_controllers_6dof.py
#
# These tests check:
#   - altitude error creates pitch command in correct direction
#   - pitch error creates elevator command
#   - airspeed error increases throttle
#   - heading error creates roll command
#   - roll error creates aileron command
#   - yaw rate creates rudder damping
#   - full autopilot returns all control commands

import numpy as np

from controllers_6dof import (
    DEFAULT_6DOF_GAINS,
    altitude_hold_controller,
    pitch_hold_controller,
    airspeed_hold_controller,
    heading_hold_controller,
    roll_hold_controller,
    yaw_damper_controller,
    full_6dof_autopilot,
)


def test_altitude_controller_direction():
    """
    If aircraft is below commanded altitude, theta_command should increase.
    """

    gains = DEFAULT_6DOF_GAINS

    theta_trim = np.deg2rad(5.0)

    theta_command = altitude_hold_controller(
        altitude_command_m=120.0,
        altitude_m=100.0,
        theta_trim_rad=theta_trim,
        gains=gains,
    )

    if theta_command <= theta_trim:
        raise AssertionError(
            "Altitude controller should command higher pitch when aircraft is below target."
        )


def test_pitch_controller_direction():
    """
    If pitch command is above actual pitch, elevator should become more negative
    than trim for nose-up correction.
    """

    gains = DEFAULT_6DOF_GAINS

    delta_e = pitch_hold_controller(
        theta_command_rad=np.deg2rad(8.0),
        theta_rad=np.deg2rad(5.0),
        q_rad_s=0.0,
        trim_elevator_rad=np.deg2rad(-2.0),
        gains=gains,
    )

    if delta_e >= np.deg2rad(-2.0):
        raise AssertionError(
            "Positive pitch error should command more negative elevator."
        )


def test_airspeed_controller_direction():
    """
    If aircraft is slower than commanded airspeed, throttle should increase.
    """

    gains = DEFAULT_6DOF_GAINS

    throttle = airspeed_hold_controller(
        airspeed_command_mps=25.0,
        airspeed_mps=23.0,
        trim_throttle=0.4,
        gains=gains,
    )

    if throttle <= 0.4:
        raise AssertionError(
            "Airspeed controller should increase throttle when aircraft is slow."
        )


def test_heading_controller_direction():
    """
    Positive heading error should create positive roll command.
    """

    gains = DEFAULT_6DOF_GAINS

    phi_command = heading_hold_controller(
        heading_command_rad=np.deg2rad(20.0),
        heading_rad=np.deg2rad(0.0),
        gains=gains,
    )

    if phi_command <= 0.0:
        raise AssertionError(
            "Positive heading error should create positive roll command."
        )


def test_roll_controller_direction():
    """
    Positive roll error should create positive aileron command.
    """

    gains = DEFAULT_6DOF_GAINS

    delta_a = roll_hold_controller(
        phi_command_rad=np.deg2rad(15.0),
        phi_rad=np.deg2rad(0.0),
        p_rad_s=0.0,
        trim_aileron_rad=0.0,
        gains=gains,
    )

    if delta_a <= 0.0:
        raise AssertionError(
            "Positive roll error should create positive aileron command."
        )


def test_yaw_damper_direction():
    """
    Positive yaw rate should create negative rudder command for damping.
    """

    gains = DEFAULT_6DOF_GAINS

    delta_r = yaw_damper_controller(
        r_rad_s=np.deg2rad(10.0),
        beta_rad=0.0,
        trim_rudder_rad=0.0,
        gains=gains,
    )

    if delta_r >= 0.0:
        raise AssertionError(
            "Positive yaw rate should create negative rudder command for damping."
        )


def test_full_autopilot_outputs():
    """
    Full autopilot should return elevator, aileron, rudder, throttle,
    and internal pitch/roll commands.
    """

    gains = DEFAULT_6DOF_GAINS

    commands = {
        "altitude_command_m": 120.0,
        "airspeed_command_mps": 25.0,
        "heading_command_rad": np.deg2rad(10.0),
    }

    state = np.array([
        0.0,          # x
        0.0,          # y
        -100.0,       # z_down
        25.0,         # u
        0.0,          # v
        0.0,          # w
        0.0,          # phi
        np.deg2rad(5.0),  # theta
        0.0,          # psi
        0.0,          # p
        0.0,          # q
        0.0,          # r
    ])

    air_data = {
        "V": 24.0,
        "alpha": np.deg2rad(5.0),
        "beta": 0.0,
    }

    trim_controls = {
        "delta_e": np.deg2rad(-2.0),
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": 0.4,
    }

    trim_info = {
        "theta_trim_rad": np.deg2rad(5.0),
    }

    controls = full_6dof_autopilot(
        commands=commands,
        state=state,
        air_data=air_data,
        trim_controls=trim_controls,
        trim_info=trim_info,
        gains=gains,
    )

    required_keys = [
        "delta_e",
        "delta_a",
        "delta_r",
        "delta_t",
        "theta_command_rad",
        "phi_command_rad",
    ]

    for key in required_keys:
        if key not in controls:
            raise AssertionError(f"Missing autopilot output key: {key}")


def main():
    print("Running 6DOF controller tests...")
    print("--------------------------------")

    test_altitude_controller_direction()
    print("Passed: altitude controller direction")

    test_pitch_controller_direction()
    print("Passed: pitch controller direction")

    test_airspeed_controller_direction()
    print("Passed: airspeed controller direction")

    test_heading_controller_direction()
    print("Passed: heading controller direction")

    test_roll_controller_direction()
    print("Passed: roll controller direction")

    test_yaw_damper_direction()
    print("Passed: yaw damper direction")

    test_full_autopilot_outputs()
    print("Passed: full autopilot outputs")

    print("\nAll 6DOF controller tests passed.")


if __name__ == "__main__":
    main()