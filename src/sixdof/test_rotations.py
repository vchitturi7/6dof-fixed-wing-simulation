# test_rotations.py
# Basic standalone tests for the 6DOF rotation utilities.
#
# Usage from project root:
#   python src/sixdof/test_rotations.py
#
# These are not full unit tests yet. They are quick sanity checks to verify:
#   - rotation matrices are orthonormal
#   - inverse rotations recover the original vector
#   - Euler rate conversion behaves correctly for simple cases
#   - heading angle wrapping works

import numpy as np

from rotations import (
    rotation_body_to_inertial,
    rotation_inertial_to_body,
    euler_rates_from_body_rates,
    wrap_angle_pi,
    test_rotation_matrix_orthonormal,
    test_rotation_inverse,
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


def test_identity_rotation():
    """
    With zero Euler angles, body frame and inertial frame are aligned.
    """

    phi = 0.0
    theta = 0.0
    psi = 0.0

    R = rotation_body_to_inertial(phi, theta, psi)

    assert_close(
        actual=R,
        expected=np.eye(3),
        tolerance=1e-12,
        message="Identity rotation failed.",
    )


def test_inverse_rotation_recovers_vector():
    """
    Rotating body -> inertial -> body should recover the original vector.
    """

    phi = np.deg2rad(15.0)
    theta = np.deg2rad(-7.0)
    psi = np.deg2rad(40.0)

    vector_body_original = np.array([25.0, 2.0, 1.0])

    R_b_to_i = rotation_body_to_inertial(phi, theta, psi)
    R_i_to_b = rotation_inertial_to_body(phi, theta, psi)

    vector_inertial = R_b_to_i @ vector_body_original
    vector_body_recovered = R_i_to_b @ vector_inertial

    assert_close(
        actual=vector_body_recovered,
        expected=vector_body_original,
        tolerance=1e-10,
        message="Inverse rotation failed to recover original vector.",
    )


def test_orthonormality():
    """
    Rotation matrix should satisfy R.T @ R = I.
    """

    phi = np.deg2rad(10.0)
    theta = np.deg2rad(5.0)
    psi = np.deg2rad(30.0)

    if not test_rotation_matrix_orthonormal(phi, theta, psi):
        raise AssertionError("Rotation matrix failed orthonormality test.")

    if not test_rotation_inverse(phi, theta, psi):
        raise AssertionError("Rotation inverse test failed.")


def test_euler_rates_level_case():
    """
    For phi = theta = 0:
        phi_dot = p
        theta_dot = q
        psi_dot = r
    """

    phi = 0.0
    theta = 0.0

    p = np.deg2rad(3.0)
    q = np.deg2rad(2.0)
    r = np.deg2rad(1.0)

    phi_dot, theta_dot, psi_dot = euler_rates_from_body_rates(
        phi=phi,
        theta=theta,
        p=p,
        q=q,
        r=r,
    )

    assert_close(
        actual=np.array([phi_dot, theta_dot, psi_dot]),
        expected=np.array([p, q, r]),
        tolerance=1e-12,
        message="Euler rates failed for level case.",
    )


def test_angle_wrapping():
    """
    190 degrees should wrap to -170 degrees.
    """

    angle = np.deg2rad(190.0)
    wrapped = wrap_angle_pi(angle)

    assert_close(
        actual=wrapped,
        expected=np.deg2rad(-170.0),
        tolerance=1e-12,
        message="Angle wrapping failed.",
    )


def main():
    print("Running rotation utility tests...")
    print("---------------------------------")

    test_identity_rotation()
    print("Passed: identity rotation")

    test_inverse_rotation_recovers_vector()
    print("Passed: inverse rotation recovers vector")

    test_orthonormality()
    print("Passed: orthonormality and inverse checks")

    test_euler_rates_level_case()
    print("Passed: Euler rates level case")

    test_angle_wrapping()
    print("Passed: angle wrapping")

    print("\nAll rotation tests passed.")


if __name__ == "__main__":
    main()