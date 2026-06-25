# aircraft_6dof_params.py
# Aircraft, aerodynamic, propulsion, and actuator parameters for the 6DOF
# fixed-wing simulation.
#
# Coordinate convention:
#   Body frame:
#       +x_body = forward through aircraft nose
#       +y_body = right wing
#       +z_body = downward through aircraft belly
#
#   Inertial frame:
#       North-East-Down style convention
#       +x_inertial = forward/north
#       +y_inertial = right/east
#       +z_inertial = down
#
# Important:
#   These are simplified small-UAV-style parameters intended for simulation,
#   controls development, and portfolio demonstration. They are not certified
#   aircraft data.

import numpy as np


# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------

G = 9.81          # gravity, m/s^2
RHO = 1.225      # air density at sea level, kg/m^3


# ---------------------------------------------------------------------
# Aircraft mass and inertia
# ---------------------------------------------------------------------

MASS = 13.5      # kg

# Moments of inertia, kg*m^2
# These are reasonable small-UAV-style values.
IXX = 0.8244
IYY = 1.135
IZZ = 1.759
IXZ = 0.1204

# For the first 6DOF implementation, we will support both:
#   1. diagonal inertia approximation
#   2. optional Ixz coupling later
USE_IXZ_COUPLING = False


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

S = 0.55         # wing reference area, m^2
B = 2.8956       # wingspan, m
C = 0.18994      # mean aerodynamic chord, m


# ---------------------------------------------------------------------
# Longitudinal aerodynamic coefficients
# ---------------------------------------------------------------------
#
# Lift:
#   CL = CL0 + CL_ALPHA*alpha + CL_Q*(c/(2V))*q + CL_DE*delta_e
#
# Drag:
#   CD = CD0 + CD_ALPHA*alpha^2 + CD_BETA*beta^2 + CD_DE*abs(delta_e)
#
# Pitch moment:
#   CM = CM0 + CM_ALPHA*alpha + CM_Q*(c/(2V))*q + CM_DE*delta_e

CL0 = 0.28
CL_ALPHA = 3.45
CL_Q = 0.0
CL_DE = -0.36

CD0 = 0.03
CD_ALPHA = 0.30
CD_BETA = 0.10
CD_DE = 0.0

CM0 = -0.02338
CM_ALPHA = -0.38
CM_Q = -3.6
CM_DE = -0.5


# ---------------------------------------------------------------------
# Lateral-directional aerodynamic coefficients
# ---------------------------------------------------------------------
#
# Side force:
#   CY = CY_BETA*beta + CY_P*(b/(2V))*p + CY_R*(b/(2V))*r
#        + CY_DA*delta_a + CY_DR*delta_r
#
# Roll moment:
#   Cl = Cl_BETA*beta + Cl_P*(b/(2V))*p + Cl_R*(b/(2V))*r
#        + Cl_DA*delta_a + Cl_DR*delta_r
#
# Yaw moment:
#   Cn = Cn_BETA*beta + Cn_P*(b/(2V))*p + Cn_R*(b/(2V))*r
#        + Cn_DA*delta_a + Cn_DR*delta_r
#
# These are simplified stability-derivative-style values. They are chosen
# to give qualitatively reasonable fixed-wing behavior:
#   - sideslip creates side force
#   - roll rate damping opposes roll rate
#   - yaw rate damping opposes yaw rate
#   - aileron creates roll moment
#   - rudder creates side force and yaw moment

CY_BETA = -0.98
CY_P = 0.0
CY_R = 0.0
CY_DA = 0.0
CY_DR = 0.17

CLL_BETA = -0.12
CLL_P = -0.26
CLL_R = 0.14
CLL_DA = 0.08
CLL_DR = 0.105

CN_BETA = 0.25
CN_P = 0.022
CN_R = -0.35
CN_DA = 0.06
CN_DR = -0.032


# ---------------------------------------------------------------------
# Propulsion
# ---------------------------------------------------------------------

MAX_THRUST = 50.0  # N

# Thrust direction in body frame.
# First model: thrust acts along +x_body.
THRUST_X_BODY = 1.0
THRUST_Y_BODY = 0.0
THRUST_Z_BODY = 0.0


# ---------------------------------------------------------------------
# Control surface limits
# ---------------------------------------------------------------------

MAX_ELEVATOR_RAD = np.deg2rad(25.0)
MAX_AILERON_RAD = np.deg2rad(25.0)
MAX_RUDDER_RAD = np.deg2rad(25.0)

MIN_THROTTLE = 0.0
MAX_THROTTLE = 1.0


# ---------------------------------------------------------------------
# Actuator dynamics
# ---------------------------------------------------------------------
#
# These match the longitudinal actuator realism idea:
#   command -> actuator lag/rate limit -> actual actuator -> aircraft dynamics

ELEVATOR_TIME_CONSTANT_S = 0.10
AILERON_TIME_CONSTANT_S = 0.10
RUDDER_TIME_CONSTANT_S = 0.10
THROTTLE_TIME_CONSTANT_S = 0.50

MAX_ELEVATOR_RATE_RAD_S = np.deg2rad(60.0)
MAX_AILERON_RATE_RAD_S = np.deg2rad(80.0)
MAX_RUDDER_RATE_RAD_S = np.deg2rad(80.0)
MAX_THROTTLE_RATE_PER_S = 1.0


# ---------------------------------------------------------------------
# Default trim / initial condition guesses
# ---------------------------------------------------------------------

DEFAULT_TRIM_AIRSPEED_MPS = 25.0
DEFAULT_TRIM_ALTITUDE_M = 100.0

DEFAULT_INITIAL_X_M = 0.0
DEFAULT_INITIAL_Y_M = 0.0
DEFAULT_INITIAL_Z_DOWN_M = -DEFAULT_TRIM_ALTITUDE_M

DEFAULT_INITIAL_PHI_RAD = 0.0
DEFAULT_INITIAL_THETA_RAD = np.deg2rad(2.0)
DEFAULT_INITIAL_PSI_RAD = 0.0

DEFAULT_INITIAL_P_RAD_S = 0.0
DEFAULT_INITIAL_Q_RAD_S = 0.0
DEFAULT_INITIAL_R_RAD_S = 0.0


# ---------------------------------------------------------------------
# Numerical protection constants
# ---------------------------------------------------------------------

MIN_AIRSPEED_MPS = 0.1
MAX_ABS_ALPHA_RAD = np.deg2rad(45.0)
MAX_ABS_BETA_RAD = np.deg2rad(30.0)