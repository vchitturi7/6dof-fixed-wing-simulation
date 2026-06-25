# rotations.py
# Rotation and kinematics utilities for the 6DOF fixed-wing simulation.
#
# Coordinate conventions:
#
# Inertial frame:
#   We use a North-East-Down style inertial frame.
#   +x_inertial = forward/north
#   +y_inertial = right/east
#   +z_inertial = down
#
# Body frame:
#   +x_body = aircraft nose direction
#   +y_body = right wing
#   +z_body = downward through aircraft belly
#
# Euler angles:
#   phi   = roll angle, rad
#   theta = pitch angle, rad
#   psi   = yaw/heading angle, rad
#
# Rotation convention:
#   R_body_to_inertial maps body-frame vectors into inertial-frame vectors.
#   R_inertial_to_body maps inertial-frame vectors into body-frame vectors.
#
# These utilities are the foundation for:
#   - 6DOF position kinematics
#   - wind transformation
#   - gravity transformation
#   - converting body velocities into inertial motion

import numpy as np


def rotation_body_to_inertial(phi, theta, psi):
    """
    Returns the direction cosine matrix that maps a vector from body frame
    to inertial frame.

    Inputs:
        phi   = roll angle, rad
        theta = pitch angle, rad
        psi   = yaw angle, rad

    Output:
        R_b_to_i = 3x3 rotation matrix

    Usage:
        vector_inertial = R_b_to_i @ vector_body
    """

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_b_to_i = np.array([
        [
            c_theta * c_psi,
            s_phi * s_theta * c_psi - c_phi * s_psi,
            c_phi * s_theta * c_psi + s_phi * s_psi,
        ],
        [
            c_theta * s_psi,
            s_phi * s_theta * s_psi + c_phi * c_psi,
            c_phi * s_theta * s_psi - s_phi * c_psi,
        ],
        [
            -s_theta,
            s_phi * c_theta,
            c_phi * c_theta,
        ],
    ])

    return R_b_to_i


def rotation_inertial_to_body(phi, theta, psi):
    """
    Returns the direction cosine matrix that maps a vector from inertial frame
    to body frame.

    Since rotation matrices are orthonormal:

        R_i_to_b = R_b_to_i.T

    Inputs:
        phi   = roll angle, rad
        theta = pitch angle, rad
        psi   = yaw angle, rad

    Output:
        R_i_to_b = 3x3 rotation matrix

    Usage:
        vector_body = R_i_to_b @ vector_inertial
    """

    return rotation_body_to_inertial(phi, theta, psi).T


def euler_rates_from_body_rates(phi, theta, p, q, r):
    """
    Converts body angular rates into Euler angle rates.

    Inputs:
        phi   = roll angle, rad
        theta = pitch angle, rad
        p     = body roll rate, rad/s
        q     = body pitch rate, rad/s
        r     = body yaw rate, rad/s

    Outputs:
        phi_dot
        theta_dot
        psi_dot

    Equations:
        phi_dot   = p + tan(theta) * (q sin(phi) + r cos(phi))
        theta_dot = q cos(phi) - r sin(phi)
        psi_dot   = (q sin(phi) + r cos(phi)) / cos(theta)

    Note:
        Euler angles become singular when theta approaches +/- 90 degrees.
        This is acceptable for the first fixed-wing sim because normal flight
        should stay far from vertical pitch attitudes.
    """

    cos_theta = np.cos(theta)

    if abs(cos_theta) < 1e-6:
        raise ValueError(
            "Euler angle singularity: cos(theta) is too close to zero. "
            "Pitch angle is near +/- 90 degrees."
        )

    phi_dot = p + np.tan(theta) * (q * np.sin(phi) + r * np.cos(phi))
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / cos_theta

    return phi_dot, theta_dot, psi_dot


def wrap_angle_pi(angle_rad):
    """
    Wraps an angle to the range [-pi, pi].

    This is useful for heading error calculations.

    Example:
        desired_heading - current_heading may produce an error larger than pi.
        Wrapping gives the shortest signed heading error.
    """

    wrapped_angle = (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    return wrapped_angle


def degrees_to_radians(angle_deg):
    """
    Converts degrees to radians.
    """

    return np.deg2rad(angle_deg)


def radians_to_degrees(angle_rad):
    """
    Converts radians to degrees.
    """

    return np.rad2deg(angle_rad)


def test_rotation_matrix_orthonormal(phi, theta, psi, tolerance=1e-9):
    """
    Checks whether the body-to-inertial rotation matrix is orthonormal.

    For a valid rotation matrix:

        R.T @ R = I

    Returns:
        True if orthonormal within tolerance, False otherwise.
    """

    R = rotation_body_to_inertial(phi, theta, psi)
    identity_check = R.T @ R

    return np.allclose(identity_check, np.eye(3), atol=tolerance)


def test_rotation_inverse(phi, theta, psi, tolerance=1e-9):
    """
    Checks whether inertial-to-body rotation is the inverse of body-to-inertial.

    For a valid pair:

        R_i_to_b @ R_b_to_i = I

    Returns:
        True if inverse relationship holds, False otherwise.
    """

    R_b_to_i = rotation_body_to_inertial(phi, theta, psi)
    R_i_to_b = rotation_inertial_to_body(phi, theta, psi)

    identity_check = R_i_to_b @ R_b_to_i

    return np.allclose(identity_check, np.eye(3), atol=tolerance)


if __name__ == "__main__":
    phi_test = np.deg2rad(10.0)
    theta_test = np.deg2rad(5.0)
    psi_test = np.deg2rad(30.0)

    print("Testing 6DOF rotation utilities...")
    print("----------------------------------")

    R_b_to_i = rotation_body_to_inertial(phi_test, theta_test, psi_test)
    R_i_to_b = rotation_inertial_to_body(phi_test, theta_test, psi_test)

    print("\nR_body_to_inertial:")
    print(R_b_to_i)

    print("\nR_inertial_to_body:")
    print(R_i_to_b)

    print("\nOrthonormal check:")
    print(test_rotation_matrix_orthonormal(phi_test, theta_test, psi_test))

    print("\nInverse check:")
    print(test_rotation_inverse(phi_test, theta_test, psi_test))

    p_test = np.deg2rad(2.0)
    q_test = np.deg2rad(1.0)
    r_test = np.deg2rad(3.0)

    phi_dot, theta_dot, psi_dot = euler_rates_from_body_rates(
        phi=phi_test,
        theta=theta_test,
        p=p_test,
        q=q_test,
        r=r_test,
    )

    print("\nEuler rates:")
    print(f"phi_dot:   {np.rad2deg(phi_dot):.6f} deg/s")
    print(f"theta_dot: {np.rad2deg(theta_dot):.6f} deg/s")
    print(f"psi_dot:   {np.rad2deg(psi_dot):.6f} deg/s")

    angle_test = np.deg2rad(190.0)
    wrapped_angle = wrap_angle_pi(angle_test)

    print("\nAngle wrapping:")
    print(f"Original: {np.rad2deg(angle_test):.3f} deg")
    print(f"Wrapped:  {np.rad2deg(wrapped_angle):.3f} deg")