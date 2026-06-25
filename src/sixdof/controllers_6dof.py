# controllers_6dof.py
# Basic controller functions for the 6DOF fixed-wing simulation.
#
# These controllers convert high-level commands into control-surface/throttle commands.
#
# Control architecture we are building toward:
#
#   Altitude command -> pitch command -> elevator
#   Airspeed command -> throttle
#   Heading command  -> roll command  -> aileron
#   Yaw damping / sideslip damping    -> rudder
#
# Coordinate/sign conventions:
#
#   Body frame:
#       +x_body = forward through aircraft nose
#       +y_body = right wing
#       +z_body = downward through aircraft belly
#
#   Euler angles:
#       phi   = roll angle, rad
#       theta = pitch angle, rad
#       psi   = heading/yaw angle, rad
#
#   Body rates:
#       p = roll rate, rad/s
#       q = pitch rate, rad/s
#       r = yaw rate, rad/s
#
# Important:
#   These are first-pass classical controllers. They are intentionally simple.
#   We will tune them after integrating them into closed-loop 6DOF simulation.

from dataclasses import dataclass
import numpy as np

import aircraft_6dof_params as p
from rotations import wrap_angle_pi


@dataclass
class Controller6DOFGains:
    """
    Gain settings for the 6DOF autopilot.

    Longitudinal:
        altitude_error -> theta_command
        theta_command - theta -> elevator
        airspeed_error -> throttle

    Lateral-directional:
        heading_error -> phi_command
        phi_command - phi -> aileron
        yaw rate / sideslip -> rudder
    """

    # Pitch hold / pitch damping
    # Pitch hold / pitch damping
    pitch_kp: float = 1.25
    pitch_kd: float = 0.50

    # Altitude outer loop
    altitude_kh_rad_per_m: float = 0.0065
    max_pitch_command_offset_rad: float = np.deg2rad(12.0)

    # Airspeed throttle loop
    airspeed_kv: float = 0.075

    # Roll hold
    roll_kp: float = 0.85
    roll_kd: float = 0.90

    # Heading outer loop
    heading_kp_rad_per_rad: float = 0.35
    max_roll_command_rad: float = np.deg2rad(18.0)

    # Rudder / yaw damping
    yaw_damper_kr: float = 0.08
    sideslip_kb: float = 0.03

    # Control limits
    max_elevator_rad: float = p.MAX_ELEVATOR_RAD
    max_aileron_rad: float = p.MAX_AILERON_RAD
    max_rudder_rad: float = p.MAX_RUDDER_RAD
    min_throttle: float = p.MIN_THROTTLE
    max_throttle: float = p.MAX_THROTTLE


DEFAULT_6DOF_GAINS = Controller6DOFGains()


def saturate(value, min_value, max_value):
    """
    Limits a scalar value between min_value and max_value.
    """

    return max(min(value, max_value), min_value)


def altitude_hold_controller(
    altitude_command_m,
    altitude_m,
    theta_trim_rad,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Outer-loop altitude controller.

    Inputs:
        altitude_command_m = commanded altitude, m
        altitude_m         = current altitude, m
        theta_trim_rad     = trim pitch angle, rad
        gains              = controller gains

    Output:
        theta_command_rad = commanded pitch angle, rad

    Logic:
        If altitude is too low, command more nose-up pitch.
        If altitude is too high, command less pitch.
    """

    altitude_error = altitude_command_m - altitude_m

    theta_command_rad = (
        theta_trim_rad
        + gains.altitude_kh_rad_per_m * altitude_error
    )

    theta_command_rad = saturate(
        theta_command_rad,
        theta_trim_rad - gains.max_pitch_command_offset_rad,
        theta_trim_rad + gains.max_pitch_command_offset_rad,
    )

    return theta_command_rad


def pitch_hold_controller(
    theta_command_rad,
    theta_rad,
    q_rad_s,
    trim_elevator_rad,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Inner-loop pitch controller.

    Inputs:
        theta_command_rad = commanded pitch angle, rad
        theta_rad         = current pitch angle, rad
        q_rad_s           = pitch rate, rad/s
        trim_elevator_rad = trim elevator deflection, rad
        gains             = controller gains

    Output:
        delta_e_rad = elevator command, rad

    Sign convention:
        The aerodynamic model uses CM_DE < 0, meaning positive elevator
        tends to produce negative pitching moment. Therefore, for a positive
        pitch error, we command more negative elevator, same as the 2D sim.
    """

    theta_error = theta_command_rad - theta_rad

    delta_e_rad = (
        trim_elevator_rad
        - gains.pitch_kp * theta_error
        + gains.pitch_kd * q_rad_s
    )

    delta_e_rad = saturate(
        delta_e_rad,
        -gains.max_elevator_rad,
        gains.max_elevator_rad,
    )

    return delta_e_rad


def airspeed_hold_controller(
    airspeed_command_mps,
    airspeed_mps,
    trim_throttle,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Airspeed hold controller.

    Inputs:
        airspeed_command_mps = commanded airspeed, m/s
        airspeed_mps         = current air-relative speed, m/s
        trim_throttle        = trim throttle, 0 to 1
        gains                = controller gains

    Output:
        delta_t = throttle command, 0 to 1

    Logic:
        If airspeed is too low, increase throttle.
        If airspeed is too high, reduce throttle.
    """

    airspeed_error = airspeed_command_mps - airspeed_mps

    delta_t = trim_throttle + gains.airspeed_kv * airspeed_error

    delta_t = saturate(
        delta_t,
        gains.min_throttle,
        gains.max_throttle,
    )

    return delta_t


def heading_hold_controller(
    heading_command_rad,
    heading_rad,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Outer-loop heading controller.

    Inputs:
        heading_command_rad = commanded heading angle, rad
        heading_rad         = current heading angle, rad
        gains               = controller gains

    Output:
        phi_command_rad = commanded roll angle, rad

    Logic:
        Heading error is wrapped to [-pi, pi] so the aircraft turns the
        shortest direction.
    """

    heading_error = wrap_angle_pi(heading_command_rad - heading_rad)

    phi_command_rad = gains.heading_kp_rad_per_rad * heading_error

    phi_command_rad = saturate(
        phi_command_rad,
        -gains.max_roll_command_rad,
        gains.max_roll_command_rad,
    )

    return phi_command_rad


def roll_hold_controller(
    phi_command_rad,
    phi_rad,
    p_rad_s,
    trim_aileron_rad=0.0,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Inner-loop roll controller.

    Inputs:
        phi_command_rad = commanded roll angle, rad
        phi_rad         = current roll angle, rad
        p_rad_s         = roll rate, rad/s
        trim_aileron_rad = trim aileron deflection, rad
        gains           = controller gains

    Output:
        delta_a_rad = aileron command, rad

    First-pass sign convention:
        Positive roll error should command positive aileron.
        Roll rate feedback damps roll motion.
    """

    roll_error = phi_command_rad - phi_rad

    delta_a_rad = (
        trim_aileron_rad
        + gains.roll_kp * roll_error
        - gains.roll_kd * p_rad_s
    )

    delta_a_rad = saturate(
        delta_a_rad,
        -gains.max_aileron_rad,
        gains.max_aileron_rad,
    )

    return delta_a_rad


def yaw_damper_controller(
    r_rad_s,
    beta_rad=0.0,
    trim_rudder_rad=0.0,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Rudder yaw damper / sideslip damper.

    Inputs:
        r_rad_s         = yaw rate, rad/s
        beta_rad        = sideslip angle, rad
        trim_rudder_rad = trim rudder deflection, rad
        gains           = controller gains

    Output:
        delta_r_rad = rudder command, rad

    First-pass logic:
        Rudder opposes yaw rate and sideslip.
    """

    delta_r_rad = (
        trim_rudder_rad
        - gains.yaw_damper_kr * r_rad_s
        - gains.sideslip_kb * beta_rad
    )

    delta_r_rad = saturate(
        delta_r_rad,
        -gains.max_rudder_rad,
        gains.max_rudder_rad,
    )

    return delta_r_rad


def full_6dof_autopilot(
    commands,
    state,
    air_data,
    trim_controls,
    trim_info,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
):
    """
    Full first-pass 6DOF autopilot.

    Inputs:
        commands dictionary:
            altitude_command_m
            airspeed_command_mps
            heading_command_rad

        state = 12-element 6DOF state vector:
            [x, y, z_down, u, v, w, phi, theta, psi, p, q, r]

        air_data dictionary:
            V
            alpha
            beta

        trim_controls dictionary:
            delta_e
            delta_a
            delta_r
            delta_t

        trim_info dictionary:
            theta_trim_rad

        gains:
            Controller6DOFGains

    Output:
        controls dictionary:
            delta_e
            delta_a
            delta_r
            delta_t
    """

    z_down = state[2]
    altitude_m = -z_down

    phi_rad = state[6]
    theta_rad = state[7]
    psi_rad = state[8]

    p_rad_s = state[9]
    q_rad_s = state[10]
    r_rad_s = state[11]

    airspeed_mps = air_data["V"]
    beta_rad = air_data["beta"]

    altitude_command_m = commands["altitude_command_m"]
    airspeed_command_mps = commands["airspeed_command_mps"]
    heading_command_rad = commands["heading_command_rad"]

    theta_trim_rad = trim_info["theta_trim_rad"]

    trim_elevator_rad = trim_controls.get("delta_e", 0.0)
    trim_aileron_rad = trim_controls.get("delta_a", 0.0)
    trim_rudder_rad = trim_controls.get("delta_r", 0.0)
    trim_throttle = trim_controls.get("delta_t", 0.0)

    theta_command_rad = altitude_hold_controller(
        altitude_command_m=altitude_command_m,
        altitude_m=altitude_m,
        theta_trim_rad=theta_trim_rad,
        gains=gains,
    )

    delta_e_rad = pitch_hold_controller(
        theta_command_rad=theta_command_rad,
        theta_rad=theta_rad,
        q_rad_s=q_rad_s,
        trim_elevator_rad=trim_elevator_rad,
        gains=gains,
    )

    delta_t = airspeed_hold_controller(
        airspeed_command_mps=airspeed_command_mps,
        airspeed_mps=airspeed_mps,
        trim_throttle=trim_throttle,
        gains=gains,
    )

    phi_command_rad = heading_hold_controller(
        heading_command_rad=heading_command_rad,
        heading_rad=psi_rad,
        gains=gains,
    )

    delta_a_rad = roll_hold_controller(
        phi_command_rad=phi_command_rad,
        phi_rad=phi_rad,
        p_rad_s=p_rad_s,
        trim_aileron_rad=trim_aileron_rad,
        gains=gains,
    )

    delta_r_rad = yaw_damper_controller(
        r_rad_s=r_rad_s,
        beta_rad=beta_rad,
        trim_rudder_rad=trim_rudder_rad,
        gains=gains,
    )

    return {
        "delta_e": delta_e_rad,
        "delta_a": delta_a_rad,
        "delta_r": delta_r_rad,
        "delta_t": delta_t,
        "theta_command_rad": theta_command_rad,
        "phi_command_rad": phi_command_rad,
    }


if __name__ == "__main__":
    gains = DEFAULT_6DOF_GAINS

    print("Default 6DOF controller gains:")
    print("------------------------------")
    print(gains)