# longitudinal_dynamics.py
# 2D longitudinal aircraft equations of motion with inertial-frame wind.
#
# State vector:
#   state[0] = u      forward body velocity, m/s
#   state[1] = w      vertical body velocity, m/s
#   state[2] = q      pitch rate, rad/s
#   state[3] = theta  pitch angle, rad
#   state[4] = h      altitude, m
#
# Control inputs:
#   controls["delta_e"] = elevator deflection, rad
#   controls["delta_t"] = throttle command, 0 to 1
#
# Optional inertial-frame wind inputs:
#   controls["wind_x_inertial"] = horizontal wind, m/s
#   controls["wind_z_inertial"] = vertical wind, m/s
#
# Inertial-frame convention:
#   +x_inertial = forward/horizontal direction
#   +z_inertial = upward
#
# Body-frame convention:
#   +x_body = forward through aircraft nose
#   +z_body = downward through aircraft belly
#
# Important:
#   Aerodynamics depend on aircraft velocity relative to the air,
#   not just aircraft velocity relative to the ground.

import numpy as np
import aircraft_params as p


def inertial_wind_to_body(wind_x_inertial, wind_z_inertial, theta):
    """
    Converts inertial-frame wind components into body-frame components.

    Inertial frame:
        +x = horizontal forward
        +z = upward

    Body frame:
        +x_body = forward along aircraft nose
        +z_body = downward through aircraft belly

    For this 2D model, theta is pitch angle.

    Returns:
        u_wind_body = wind component along body x, m/s
        w_wind_body = wind component along body z, m/s
    """

    # Inertial wind vector using x-forward, z-up convention
    wind_inertial = np.array([
        wind_x_inertial,
        wind_z_inertial,
    ])

    # Body axes expressed in inertial coordinates:
    # body x-axis points along aircraft nose
    # body z-axis points downward relative to aircraft
    body_x_inertial = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    body_z_inertial = np.array([
        -np.sin(theta),
        -np.cos(theta),
    ])

    u_wind_body = np.dot(wind_inertial, body_x_inertial)
    w_wind_body = np.dot(wind_inertial, body_z_inertial)

    return u_wind_body, w_wind_body


def longitudinal_dynamics(t, state, controls):
    """
    Computes 2D longitudinal aircraft dynamics.

    The aircraft state variables u and w are body-axis velocities.
    Aerodynamic forces are calculated using the velocity of the aircraft
    relative to the moving air mass.

    Wind is defined in the inertial frame and converted into the body frame.
    """

    u, w, q, theta, h = state

    delta_e = controls["delta_e"]
    delta_t = controls["delta_t"]

    # Inertial-frame wind inputs
    wind_x_inertial = controls.get("wind_x_inertial", 0.0)
    wind_z_inertial = controls.get("wind_z_inertial", 0.0)

    # Convert inertial wind to body-frame wind
    u_wind_body, w_wind_body = inertial_wind_to_body(
        wind_x_inertial=wind_x_inertial,
        wind_z_inertial=wind_z_inertial,
        theta=theta,
    )

    # Relative airflow seen by aircraft
    u_rel = u - u_wind_body
    w_rel = w - w_wind_body

    # Avoid divide-by-zero issues
    V = np.sqrt(u_rel**2 + w_rel**2)
    V = max(V, 0.1)

    # Angle of attack based on relative wind
    alpha = np.arctan2(w_rel, u_rel)

    # Dynamic pressure
    q_bar = 0.5 * p.RHO * V**2

    # Aerodynamic coefficients
    CL = (
        p.CL0
        + p.CL_ALPHA * alpha
        + p.CL_Q * (p.C / (2 * V)) * q
        + p.CL_DE * delta_e
    )

    CD = (
        p.CD0
        + p.CD_ALPHA * alpha**2
        + p.CD_DE * abs(delta_e)
    )

    CM = (
        p.CM0
        + p.CM_ALPHA * alpha
        + p.CM_Q * (p.C / (2 * V)) * q
        + p.CM_DE * delta_e
    )

    # Lift, drag, and pitching moment
    L = q_bar * p.S * CL
    D = q_bar * p.S * CD
    M = q_bar * p.S * p.C * CM

    # Convert lift/drag from wind axes to body axes
    # X is forward body force
    # Z is downward body force
    X_aero = -D * np.cos(alpha) + L * np.sin(alpha)
    Z_aero = -D * np.sin(alpha) - L * np.cos(alpha)

    # Thrust acts in body x direction
    T = p.MAX_THRUST * delta_t

    X = X_aero + T
    Z = Z_aero

    # Longitudinal equations of motion
    u_dot = X / p.MASS - p.G * np.sin(theta) - q * w
    w_dot = Z / p.MASS + p.G * np.cos(theta) + q * u
    q_dot = M / p.IYY
    theta_dot = q

    # Altitude rate
    # h is positive upward
    h_dot = u * np.sin(theta) - w * np.cos(theta)

    return np.array([u_dot, w_dot, q_dot, theta_dot, h_dot])