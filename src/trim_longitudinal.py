
import numpy as np
from scipy.optimize import root

import aircraft_params as p
from longitudinal_dynamics import longitudinal_dynamics


def trim_residual(unknowns, target_airspeed):
    """
    Residual equations for steady level flight.

    Unknowns:
        alpha   = angle of attack, rad
        theta   = pitch angle, rad
        delta_e = elevator deflection, rad
        delta_t = throttle command, 0 to 1

    We want:
        u_dot = 0
        w_dot = 0
        q_dot = 0
        h_dot = 0
    """

    alpha, theta, delta_e, delta_t = unknowns

    # Convert target airspeed and alpha into body-frame velocities
    u = target_airspeed * np.cos(alpha)
    w = target_airspeed * np.sin(alpha)

    q = 0.0
    h = p.H0

    state = np.array([u, w, q, theta, h])

    controls = {
        "delta_e": delta_e,
        "delta_t": delta_t,
    }

    state_dot = longitudinal_dynamics(0.0, state, controls)

    u_dot = state_dot[0]
    w_dot = state_dot[1]
    q_dot = state_dot[2]
    h_dot = state_dot[4]

    return np.array([u_dot, w_dot, q_dot, h_dot])


def solve_trim(target_airspeed=25.0):
    """
    Solves for alpha, theta, elevator, and throttle for steady level flight.
    """

    # Initial guess:
    # alpha = 3 deg, theta = 3 deg, elevator = 0 deg, throttle = 50%
    initial_guess = np.array([
        np.deg2rad(3.0),
        np.deg2rad(3.0),
        np.deg2rad(0.0),
        0.5,
    ])

    solution = root(
        trim_residual,
        initial_guess,
        args=(target_airspeed,),
        method="hybr",
    )

    if not solution.success:
        raise RuntimeError(f"Trim solver failed: {solution.message}")

    alpha_trim, theta_trim, delta_e_trim, delta_t_trim = solution.x

    # Keep throttle physically reasonable
    if delta_t_trim < 0.0 or delta_t_trim > 1.0:
        print("WARNING: Trim throttle is outside [0, 1].")
        print("This may mean the aircraft parameters/thrust model need adjustment.")

    u_trim = target_airspeed * np.cos(alpha_trim)
    w_trim = target_airspeed * np.sin(alpha_trim)
    q_trim = 0.0
    h_trim = p.H0

    trim_state = np.array([
        u_trim,
        w_trim,
        q_trim,
        theta_trim,
        h_trim,
    ])

    trim_controls = {
        "delta_e": delta_e_trim,
        "delta_t": delta_t_trim,
    }

    trim_info = {
        "alpha_trim_rad": alpha_trim,
        "alpha_trim_deg": np.rad2deg(alpha_trim),
        "theta_trim_rad": theta_trim,
        "theta_trim_deg": np.rad2deg(theta_trim),
        "delta_e_trim_rad": delta_e_trim,
        "delta_e_trim_deg": np.rad2deg(delta_e_trim),
        "delta_t_trim": delta_t_trim,
        "u_trim": u_trim,
        "w_trim": w_trim,
        "target_airspeed": target_airspeed,
    }

    return trim_state, trim_controls, trim_info


if __name__ == "__main__":
    trim_state, trim_controls, trim_info = solve_trim(target_airspeed=25.0)

    print("\nTrim solution:")
    for key, value in trim_info.items():
        print(f"{key}: {value}")

    print("\nTrim state:")
    print(trim_state)

    print("\nTrim controls:")
    print(trim_controls)

    print("\nState derivatives at trim:")
    print(longitudinal_dynamics(0.0, trim_state, trim_controls))