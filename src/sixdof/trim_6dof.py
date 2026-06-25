# trim_6dof.py
# Trim solver for straight and level 6DOF fixed-wing flight.
#
# This solver finds:
#   alpha
#   theta
#   elevator
#   throttle
#
# for a target airspeed and altitude such that the aircraft is approximately
# in steady, level, wings-level flight.
#
# Assumptions for first 6DOF trim:
#   phi = 0
#   psi = 0
#   v = 0
#   p = q = r = 0
#   delta_a = 0
#   delta_r = 0
#
# Residual targets:
#   u_dot = 0
#   w_dot = 0
#   q_dot = 0
#   z_down_dot = 0
#
# z_down_dot = 0 means no climb/descent, because altitude = -z_down.

import numpy as np
from scipy.optimize import root

import aircraft_6dof_params as p
from dynamics_6dof import dynamics_6dof


def build_trim_state(
    target_airspeed_mps,
    altitude_m,
    alpha_rad,
    theta_rad,
):
    """
    Builds a 12-state 6DOF vector from trim variables.

    Inputs:
        target_airspeed_mps = desired airspeed, m/s
        altitude_m          = desired altitude, m
        alpha_rad           = angle of attack, rad
        theta_rad           = pitch angle, rad

    Returns:
        state = 12-element 6DOF state vector
    """

    u = target_airspeed_mps * np.cos(alpha_rad)
    v = 0.0
    w = target_airspeed_mps * np.sin(alpha_rad)

    x_inertial = 0.0
    y_inertial = 0.0
    z_down = -altitude_m

    phi = 0.0
    theta = theta_rad
    psi = 0.0

    p_rate = 0.0
    q_rate = 0.0
    r_rate = 0.0

    state = np.array([
        x_inertial,
        y_inertial,
        z_down,
        u,
        v,
        w,
        phi,
        theta,
        psi,
        p_rate,
        q_rate,
        r_rate,
    ])

    return state


def build_trim_controls(delta_e_rad, delta_t):
    """
    Builds control dictionary for trim.

    Inputs:
        delta_e_rad = elevator deflection, rad
        delta_t     = throttle command, 0 to 1

    Returns:
        controls dictionary
    """

    return {
        "delta_e": delta_e_rad,
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": delta_t,
        "wind_inertial": np.zeros(3),
    }


def trim_residual(unknowns, target_airspeed_mps, altitude_m):
    """
    Residual equations for straight and level trim.

    Unknowns:
        alpha_rad
        theta_rad
        delta_e_rad
        delta_t

    Residuals:
        u_dot
        w_dot
        q_dot
        z_down_dot
    """

    alpha_rad, theta_rad, delta_e_rad, delta_t = unknowns

    state = build_trim_state(
        target_airspeed_mps=target_airspeed_mps,
        altitude_m=altitude_m,
        alpha_rad=alpha_rad,
        theta_rad=theta_rad,
    )

    controls = build_trim_controls(
        delta_e_rad=delta_e_rad,
        delta_t=delta_t,
    )

    state_dot = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls,
    )

    u_dot = state_dot[3]
    w_dot = state_dot[5]
    q_dot = state_dot[10]
    z_down_dot = state_dot[2]

    return np.array([
        u_dot,
        w_dot,
        q_dot,
        z_down_dot,
    ])


def solve_trim_6dof(
    target_airspeed_mps=25.0,
    altitude_m=100.0,
):
    """
    Solves for 6DOF straight and level trim.

    Returns:
        trim_state
        trim_controls
        trim_info
    """

    initial_guess = np.array([
        np.deg2rad(3.0),   # alpha
        np.deg2rad(3.0),   # theta
        np.deg2rad(0.0),   # elevator
        0.5,               # throttle
    ])

    solution = root(
        trim_residual,
        initial_guess,
        args=(target_airspeed_mps, altitude_m),
        method="hybr",
    )

    if not solution.success:
        raise RuntimeError(f"6DOF trim solver failed: {solution.message}")

    alpha_trim, theta_trim, delta_e_trim, delta_t_trim = solution.x

    if delta_t_trim < 0.0 or delta_t_trim > 1.0:
        print("WARNING: 6DOF trim throttle is outside [0, 1].")
        print("This may indicate the simplified aircraft model needs tuning.")

    trim_state = build_trim_state(
        target_airspeed_mps=target_airspeed_mps,
        altitude_m=altitude_m,
        alpha_rad=alpha_trim,
        theta_rad=theta_trim,
    )

    trim_controls = build_trim_controls(
        delta_e_rad=delta_e_trim,
        delta_t=delta_t_trim,
    )

    trim_info = {
        "target_airspeed_mps": target_airspeed_mps,
        "altitude_m": altitude_m,
        "alpha_trim_rad": alpha_trim,
        "alpha_trim_deg": np.rad2deg(alpha_trim),
        "theta_trim_rad": theta_trim,
        "theta_trim_deg": np.rad2deg(theta_trim),
        "delta_e_trim_rad": delta_e_trim,
        "delta_e_trim_deg": np.rad2deg(delta_e_trim),
        "delta_t_trim": delta_t_trim,
        "u_trim": trim_state[3],
        "v_trim": trim_state[4],
        "w_trim": trim_state[5],
        "phi_trim_rad": trim_state[6],
        "psi_trim_rad": trim_state[8],
    }

    return trim_state, trim_controls, trim_info


if __name__ == "__main__":
    trim_state, trim_controls, trim_info = solve_trim_6dof(
        target_airspeed_mps=p.DEFAULT_TRIM_AIRSPEED_MPS,
        altitude_m=p.DEFAULT_TRIM_ALTITUDE_M,
    )

    print("6DOF trim solution")
    print("------------------")

    for key, value in trim_info.items():
        print(f"{key}: {value}")

    print("\nTrim state:")
    print(trim_state)

    print("\nTrim controls:")
    print(trim_controls)

    print("\nState derivatives at trim:")
    print(
        dynamics_6dof(
            t=0.0,
            state=trim_state,
            controls=trim_controls,
        )
    )