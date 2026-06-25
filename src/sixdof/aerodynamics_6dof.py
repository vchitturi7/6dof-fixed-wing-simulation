# aerodynamics_6dof.py
# Aerodynamic force and moment model for the 6DOF fixed-wing simulation.
#
# Coordinate convention:
#   Body frame:
#       +x_body = forward
#       +y_body = right wing
#       +z_body = downward
#
# In this convention:
#   X force positive forward
#   Y force positive right
#   Z force positive downward
#   L moment positive roll right
#   M moment positive nose-up/nose-down according to standard body-axis moment sign
#   N moment positive yaw right
#
# Aerodynamic model:
#   1. Compute relative velocity in body axes.
#   2. Compute airspeed V, angle of attack alpha, and sideslip beta.
#   3. Compute aerodynamic coefficients.
#   4. Convert lift/drag/side force into body-axis forces.
#   5. Compute roll, pitch, yaw moments.

import numpy as np

import aircraft_6dof_params as p


def saturate(value, min_value, max_value):
    """
    Limits a scalar value between min_value and max_value.
    """

    return max(min(value, max_value), min_value)


def compute_air_data(velocity_body, wind_body=None):
    """
    Computes relative airspeed, angle of attack, and sideslip.

    Inputs:
        velocity_body = np.array([u, v, w]), m/s
        wind_body     = np.array([wind_u, wind_v, wind_w]), m/s

    Returns:
        air_data dictionary containing:
            u_rel
            v_rel
            w_rel
            V
            alpha
            beta

    Notes:
        velocity_relative_body = aircraft_body_velocity - wind_body

        alpha = atan2(w_rel, u_rel)
        beta  = asin(v_rel / V)

    The signs are consistent with:
        +u = forward
        +v = right
        +w = down
    """

    if wind_body is None:
        wind_body = np.zeros(3)

    u, v, w = velocity_body
    wind_u, wind_v, wind_w = wind_body

    u_rel = u - wind_u
    v_rel = v - wind_v
    w_rel = w - wind_w

    V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
    V = max(V, p.MIN_AIRSPEED_MPS)

    alpha = np.arctan2(w_rel, u_rel)
    beta = np.arcsin(saturate(v_rel / V, -1.0, 1.0))

    alpha = saturate(
        alpha,
        -p.MAX_ABS_ALPHA_RAD,
        p.MAX_ABS_ALPHA_RAD,
    )

    beta = saturate(
        beta,
        -p.MAX_ABS_BETA_RAD,
        p.MAX_ABS_BETA_RAD,
    )

    return {
        "u_rel": u_rel,
        "v_rel": v_rel,
        "w_rel": w_rel,
        "V": V,
        "alpha": alpha,
        "beta": beta,
    }


def compute_aero_coefficients(
    alpha,
    beta,
    p_rate,
    q_rate,
    r_rate,
    controls,
    airspeed,
):
    """
    Computes aerodynamic coefficients.

    Inputs:
        alpha   = angle of attack, rad
        beta    = sideslip angle, rad
        p_rate  = roll rate, rad/s
        q_rate  = pitch rate, rad/s
        r_rate  = yaw rate, rad/s
        controls dictionary:
            delta_e = elevator, rad
            delta_a = aileron, rad
            delta_r = rudder, rad
        airspeed = V, m/s

    Returns:
        dictionary of aerodynamic coefficients:
            CL, CD, CY, Cl, Cm, Cn
    """

    V = max(airspeed, p.MIN_AIRSPEED_MPS)

    delta_e = controls.get("delta_e", 0.0)
    delta_a = controls.get("delta_a", 0.0)
    delta_r = controls.get("delta_r", 0.0)

    delta_e = saturate(delta_e, -p.MAX_ELEVATOR_RAD, p.MAX_ELEVATOR_RAD)
    delta_a = saturate(delta_a, -p.MAX_AILERON_RAD, p.MAX_AILERON_RAD)
    delta_r = saturate(delta_r, -p.MAX_RUDDER_RAD, p.MAX_RUDDER_RAD)

    p_hat = p.B / (2.0 * V) * p_rate
    q_hat = p.C / (2.0 * V) * q_rate
    r_hat = p.B / (2.0 * V) * r_rate

    CL = (
        p.CL0
        + p.CL_ALPHA * alpha
        + p.CL_Q * q_hat
        + p.CL_DE * delta_e
    )

    CD = (
        p.CD0
        + p.CD_ALPHA * alpha**2
        + p.CD_BETA * beta**2
        + p.CD_DE * abs(delta_e)
    )

    CY = (
        p.CY_BETA * beta
        + p.CY_P * p_hat
        + p.CY_R * r_hat
        + p.CY_DA * delta_a
        + p.CY_DR * delta_r
    )

    Cl = (
        p.CLL_BETA * beta
        + p.CLL_P * p_hat
        + p.CLL_R * r_hat
        + p.CLL_DA * delta_a
        + p.CLL_DR * delta_r
    )

    Cm = (
        p.CM0
        + p.CM_ALPHA * alpha
        + p.CM_Q * q_hat
        + p.CM_DE * delta_e
    )

    Cn = (
        p.CN_BETA * beta
        + p.CN_P * p_hat
        + p.CN_R * r_hat
        + p.CN_DA * delta_a
        + p.CN_DR * delta_r
    )

    return {
        "CL": CL,
        "CD": CD,
        "CY": CY,
        "Cl": Cl,
        "Cm": Cm,
        "Cn": Cn,
    }


def aerodynamic_forces_moments(
    velocity_body,
    angular_rates_body,
    controls,
    wind_body=None,
):
    """
    Computes aerodynamic forces and moments in body axes.

    Inputs:
        velocity_body = np.array([u, v, w]), m/s
        angular_rates_body = np.array([p, q, r]), rad/s
        controls dictionary:
            delta_e = elevator, rad
            delta_a = aileron, rad
            delta_r = rudder, rad
        wind_body = np.array([wind_u, wind_v, wind_w]), m/s

    Returns:
        aero_output dictionary:
            forces_body = np.array([X, Y, Z]), N
            moments_body = np.array([L, M, N]), N*m
            coefficients = dictionary of aero coefficients
            air_data = dictionary with V, alpha, beta, etc.
    """

    p_rate, q_rate, r_rate = angular_rates_body

    air_data = compute_air_data(
        velocity_body=velocity_body,
        wind_body=wind_body,
    )

    V = air_data["V"]
    alpha = air_data["alpha"]
    beta = air_data["beta"]

    coefficients = compute_aero_coefficients(
        alpha=alpha,
        beta=beta,
        p_rate=p_rate,
        q_rate=q_rate,
        r_rate=r_rate,
        controls=controls,
        airspeed=V,
    )

    CL = coefficients["CL"]
    CD = coefficients["CD"]
    CY = coefficients["CY"]
    Cl = coefficients["Cl"]
    Cm = coefficients["Cm"]
    Cn = coefficients["Cn"]

    q_bar = 0.5 * p.RHO * V**2

    lift = q_bar * p.S * CL
    drag = q_bar * p.S * CD
    side_force = q_bar * p.S * CY

    # Convert lift and drag to body-axis X/Z forces.
    # This matches the existing 2D longitudinal convention:
    #   X positive forward
    #   Z positive downward
    X_aero = -drag * np.cos(alpha) + lift * np.sin(alpha)
    Z_aero = -drag * np.sin(alpha) - lift * np.cos(alpha)

    # First lateral model:
    #   Y positive right.
    Y_aero = side_force

    roll_moment = q_bar * p.S * p.B * Cl
    pitch_moment = q_bar * p.S * p.C * Cm
    yaw_moment = q_bar * p.S * p.B * Cn

    forces_body = np.array([
        X_aero,
        Y_aero,
        Z_aero,
    ])

    moments_body = np.array([
        roll_moment,
        pitch_moment,
        yaw_moment,
    ])

    return {
        "forces_body": forces_body,
        "moments_body": moments_body,
        "coefficients": coefficients,
        "air_data": air_data,
        "dynamic_pressure": q_bar,
        "lift": lift,
        "drag": drag,
        "side_force": side_force,
    }


def propulsion_forces_moments(throttle):
    """
    Computes propulsion forces and moments.

    Inputs:
        throttle = throttle command/actual value from 0 to 1

    Returns:
        propulsion_output dictionary:
            forces_body = np.array([X, Y, Z]), N
            moments_body = np.array([L, M, N]), N*m

    First model:
        Thrust acts along +x_body through the center of mass.
        Therefore, thrust creates no moment.
    """

    throttle = saturate(
        throttle,
        p.MIN_THROTTLE,
        p.MAX_THROTTLE,
    )

    thrust = p.MAX_THRUST * throttle

    forces_body = thrust * np.array([
        p.THRUST_X_BODY,
        p.THRUST_Y_BODY,
        p.THRUST_Z_BODY,
    ])

    moments_body = np.array([
        0.0,
        0.0,
        0.0,
    ])

    return {
        "forces_body": forces_body,
        "moments_body": moments_body,
        "thrust": thrust,
    }


def total_forces_moments(
    velocity_body,
    angular_rates_body,
    controls,
    wind_body=None,
):
    """
    Computes total non-gravity forces and moments in body axes.

    This includes:
        aerodynamic forces/moments
        propulsion forces/moments

    Gravity is handled separately in the 6DOF dynamics file because gravity
    must be transformed based on aircraft attitude.

    Inputs:
        velocity_body = np.array([u, v, w]), m/s
        angular_rates_body = np.array([p, q, r]), rad/s
        controls:
            delta_e
            delta_a
            delta_r
            delta_t
        wind_body = np.array([wind_u, wind_v, wind_w]), m/s

    Returns:
        output dictionary:
            forces_body
            moments_body
            aero
            propulsion
    """

    aero = aerodynamic_forces_moments(
        velocity_body=velocity_body,
        angular_rates_body=angular_rates_body,
        controls=controls,
        wind_body=wind_body,
    )

    propulsion = propulsion_forces_moments(
        throttle=controls.get("delta_t", 0.0),
    )

    forces_body = aero["forces_body"] + propulsion["forces_body"]
    moments_body = aero["moments_body"] + propulsion["moments_body"]

    return {
        "forces_body": forces_body,
        "moments_body": moments_body,
        "aero": aero,
        "propulsion": propulsion,
    }


if __name__ == "__main__":
    velocity_body_test = np.array([25.0, 0.0, 1.0])
    angular_rates_test = np.array([0.0, 0.0, 0.0])

    controls_test = {
        "delta_e": np.deg2rad(0.0),
        "delta_a": np.deg2rad(0.0),
        "delta_r": np.deg2rad(0.0),
        "delta_t": 0.5,
    }

    output = total_forces_moments(
        velocity_body=velocity_body_test,
        angular_rates_body=angular_rates_test,
        controls=controls_test,
        wind_body=np.zeros(3),
    )

    print("6DOF aero/propulsion test output")
    print("--------------------------------")
    print(f"Forces body [N]:  {output['forces_body']}")
    print(f"Moments body [Nm]: {output['moments_body']}")
    print(f"Air data:          {output['aero']['air_data']}")
    print(f"Coefficients:      {output['aero']['coefficients']}")
    print(f"Thrust [N]:        {output['propulsion']['thrust']:.3f}")