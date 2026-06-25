# dynamics_6dof.py
# Full 6DOF rigid-body dynamics for the fixed-wing simulation.
#
# State vector:
#   state[0]  = x_inertial      position x/north, m
#   state[1]  = y_inertial      position y/east, m
#   state[2]  = z_down          position down, m
#   state[3]  = u               body x velocity, m/s
#   state[4]  = v               body y velocity, m/s
#   state[5]  = w               body z velocity, m/s
#   state[6]  = phi             roll angle, rad
#   state[7]  = theta           pitch angle, rad
#   state[8]  = psi             yaw/heading angle, rad
#   state[9]  = p               roll rate, rad/s
#   state[10] = q               pitch rate, rad/s
#   state[11] = r               yaw rate, rad/s
#
# Control dictionary:
#   controls["delta_e"] = elevator deflection, rad
#   controls["delta_a"] = aileron deflection, rad
#   controls["delta_r"] = rudder deflection, rad
#   controls["delta_t"] = throttle, 0 to 1
#
# Optional wind inputs:
#   controls["wind_inertial"] = np.array([wind_x, wind_y, wind_z_down]), m/s
#
# Coordinate convention:
#   Inertial frame is North-East-Down style:
#       +x_inertial = forward/north
#       +y_inertial = right/east
#       +z_inertial = down
#
#   Body frame:
#       +x_body = forward through aircraft nose
#       +y_body = right wing
#       +z_body = downward through aircraft belly
#
# Important:
#   Gravity is positive in +z_down inertial direction.
#   Aerodynamics and propulsion are computed in body axes.

import numpy as np

import aircraft_6dof_params as p
from rotations import (
    rotation_body_to_inertial,
    rotation_inertial_to_body,
    euler_rates_from_body_rates,
)
from aerodynamics_6dof import total_forces_moments


def unpack_state(state):
    """
    Unpacks the 12-state vector into named components.
    """

    x_inertial = state[0]
    y_inertial = state[1]
    z_down = state[2]

    u = state[3]
    v = state[4]
    w = state[5]

    phi = state[6]
    theta = state[7]
    psi = state[8]

    p_rate = state[9]
    q_rate = state[10]
    r_rate = state[11]

    return {
        "x_inertial": x_inertial,
        "y_inertial": y_inertial,
        "z_down": z_down,
        "u": u,
        "v": v,
        "w": w,
        "phi": phi,
        "theta": theta,
        "psi": psi,
        "p_rate": p_rate,
        "q_rate": q_rate,
        "r_rate": r_rate,
    }


def pack_state_derivative(
    position_dot_inertial,
    velocity_dot_body,
    euler_dot,
    angular_rate_dot_body,
):
    """
    Packs derivatives into a 12-element state derivative vector.
    """

    return np.array([
        position_dot_inertial[0],
        position_dot_inertial[1],
        position_dot_inertial[2],
        velocity_dot_body[0],
        velocity_dot_body[1],
        velocity_dot_body[2],
        euler_dot[0],
        euler_dot[1],
        euler_dot[2],
        angular_rate_dot_body[0],
        angular_rate_dot_body[1],
        angular_rate_dot_body[2],
    ])


def gravity_body(phi, theta, psi):
    """
    Computes gravity acceleration vector expressed in body axes.

    In inertial NED coordinates:
        gravity_inertial = [0, 0, +g]

    Because +z_down is positive downward.

    Returns:
        gravity_body_vector = np.array([gx_body, gy_body, gz_body]), m/s^2
    """

    gravity_inertial = np.array([
        0.0,
        0.0,
        p.G,
    ])

    R_i_to_b = rotation_inertial_to_body(phi, theta, psi)

    return R_i_to_b @ gravity_inertial


def wind_body_from_controls(phi, theta, psi, controls):
    """
    Converts inertial-frame wind from controls into body-frame wind.

    controls may contain:
        controls["wind_inertial"] = np.array([wind_x, wind_y, wind_z_down])

    If no wind is supplied, returns zero wind.
    """

    wind_inertial = controls.get("wind_inertial", np.zeros(3))

    wind_inertial = np.array(wind_inertial, dtype=float)

    R_i_to_b = rotation_inertial_to_body(phi, theta, psi)

    return R_i_to_b @ wind_inertial


def translational_dynamics_body(
    velocity_body,
    angular_rates_body,
    forces_body,
    gravity_body_vector,
):
    """
    Computes body-axis translational acceleration.

    Equation:
        v_dot_body = F_body / m + gravity_body - omega_body x v_body

    Expanded:
        u_dot = Fx/m + gx + r*v - q*w
        v_dot = Fy/m + gy + p*w - r*u
        w_dot = Fz/m + gz + q*u - p*v
    """

    u, v, w = velocity_body
    p_rate, q_rate, r_rate = angular_rates_body

    force_accel = forces_body / p.MASS

    rotational_coupling = np.array([
        r_rate * v - q_rate * w,
        p_rate * w - r_rate * u,
        q_rate * u - p_rate * v,
    ])

    velocity_dot_body = force_accel + gravity_body_vector + rotational_coupling

    return velocity_dot_body


def rotational_dynamics_body(angular_rates_body, moments_body):
    """
    Computes body-axis angular acceleration.

    First implementation uses diagonal inertia:
        I = diag(Ixx, Iyy, Izz)

    Equations:
        p_dot = (L + (Iyy - Izz) q r) / Ixx
        q_dot = (M + (Izz - Ixx) p r) / Iyy
        r_dot = (N + (Ixx - Iyy) p q) / Izz

    The aircraft parameter file includes Ixz, but Ixz coupling is intentionally
    disabled for the first 6DOF implementation.
    """

    p_rate, q_rate, r_rate = angular_rates_body
    roll_moment, pitch_moment, yaw_moment = moments_body

    if p.USE_IXZ_COUPLING:
        raise NotImplementedError(
            "IXZ inertia coupling is not implemented yet. "
            "Set USE_IXZ_COUPLING = False in aircraft_6dof_params.py."
        )

    p_dot = (
        roll_moment
        + (p.IYY - p.IZZ) * q_rate * r_rate
    ) / p.IXX

    q_dot = (
        pitch_moment
        + (p.IZZ - p.IXX) * p_rate * r_rate
    ) / p.IYY

    r_dot = (
        yaw_moment
        + (p.IXX - p.IYY) * p_rate * q_rate
    ) / p.IZZ

    angular_rate_dot_body = np.array([
        p_dot,
        q_dot,
        r_dot,
    ])

    return angular_rate_dot_body


def dynamics_6dof(t, state, controls):
    """
    Computes full 6DOF rigid-body aircraft dynamics.

    Inputs:
        t        = time, s
        state    = 12-element state vector
        controls = control dictionary

    Returns:
        state_dot = 12-element derivative vector
    """

    state_parts = unpack_state(state)

    u = state_parts["u"]
    v = state_parts["v"]
    w = state_parts["w"]

    phi = state_parts["phi"]
    theta = state_parts["theta"]
    psi = state_parts["psi"]

    p_rate = state_parts["p_rate"]
    q_rate = state_parts["q_rate"]
    r_rate = state_parts["r_rate"]

    velocity_body = np.array([
        u,
        v,
        w,
    ])

    angular_rates_body = np.array([
        p_rate,
        q_rate,
        r_rate,
    ])

    R_b_to_i = rotation_body_to_inertial(phi, theta, psi)

    # Position kinematics.
    position_dot_inertial = R_b_to_i @ velocity_body

    # Wind transformation.
    wind_body = wind_body_from_controls(
        phi=phi,
        theta=theta,
        psi=psi,
        controls=controls,
    )

    # Non-gravity forces and moments.
    forces_moments = total_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls,
        wind_body=wind_body,
    )

    forces_body = forces_moments["forces_body"]
    moments_body = forces_moments["moments_body"]

    # Gravity in body frame.
    gravity_body_vector = gravity_body(
        phi=phi,
        theta=theta,
        psi=psi,
    )

    # Translational dynamics.
    velocity_dot_body = translational_dynamics_body(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        forces_body=forces_body,
        gravity_body_vector=gravity_body_vector,
    )

    # Attitude kinematics.
    phi_dot, theta_dot, psi_dot = euler_rates_from_body_rates(
        phi=phi,
        theta=theta,
        p=p_rate,
        q=q_rate,
        r=r_rate,
    )

    euler_dot = np.array([
        phi_dot,
        theta_dot,
        psi_dot,
    ])

    # Rotational dynamics.
    angular_rate_dot_body = rotational_dynamics_body(
        angular_rates_body=angular_rates_body,
        moments_body=moments_body,
    )

    state_dot = pack_state_derivative(
        position_dot_inertial=position_dot_inertial,
        velocity_dot_body=velocity_dot_body,
        euler_dot=euler_dot,
        angular_rate_dot_body=angular_rate_dot_body,
    )

    return state_dot


def make_initial_state(
    airspeed_mps=p.DEFAULT_TRIM_AIRSPEED_MPS,
    altitude_m=p.DEFAULT_TRIM_ALTITUDE_M,
    phi_rad=p.DEFAULT_INITIAL_PHI_RAD,
    theta_rad=p.DEFAULT_INITIAL_THETA_RAD,
    psi_rad=p.DEFAULT_INITIAL_PSI_RAD,
):
    """
    Creates a simple initial 6DOF state.

    This is not a trimmed state yet.
    It is only a convenient starting point for sanity tests.

    State:
        position = [0, 0, -altitude]
        velocity_body = [airspeed, 0, 0]
        attitude = [phi, theta, psi]
        rates = [0, 0, 0]
    """

    return np.array([
        p.DEFAULT_INITIAL_X_M,
        p.DEFAULT_INITIAL_Y_M,
        -altitude_m,
        airspeed_mps,
        0.0,
        0.0,
        phi_rad,
        theta_rad,
        psi_rad,
        0.0,
        0.0,
        0.0,
    ])


if __name__ == "__main__":
    state = make_initial_state()

    controls = {
        "delta_e": 0.0,
        "delta_a": 0.0,
        "delta_r": 0.0,
        "delta_t": 0.5,
        "wind_inertial": np.zeros(3),
    }

    state_dot = dynamics_6dof(
        t=0.0,
        state=state,
        controls=controls,
    )

    print("6DOF dynamics sanity run")
    print("-----------------------")
    print("Initial state:")
    print(state)
    print("\nState derivative:")
    print(state_dot)