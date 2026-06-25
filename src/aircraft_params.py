# aircraft_params.py
# Basic small fixed-wing UAV parameters for a first longitudinal simulation.

import numpy as np

# Physical constants
G = 9.81          # gravity, m/s^2
RHO = 1.225      # air density at sea level, kg/m^3

# Aircraft properties
MASS = 13.5      # kg
IYY = 1.135      # kg*m^2, pitch-axis moment of inertia

# Geometry
S = 0.55         # wing area, m^2
C = 0.18994      # mean aerodynamic chord, m

# Aerodynamic coefficients
CL0 = 0.28
CL_ALPHA = 3.45
CL_Q = 0.0
CL_DE = -0.36

CD0 = 0.03
CD_ALPHA = 0.30
CD_DE = 0.0

CM0 = -0.02338
CM_ALPHA = -0.38
CM_Q = -3.6
CM_DE = -0.5

# Propulsion
MAX_THRUST = 50.0  # N

# Initial condition
U0 = 25.0                  # initial forward velocity, m/s
W0 = 0.0                   # initial vertical body velocity, m/s
Q0 = 0.0                   # initial pitch rate, rad/s
THETA0 = np.deg2rad(2.0)   # initial pitch angle, rad
H0 = 100.0                 # initial altitude, m

# Actuator dynamics
ELEVATOR_TIME_CONSTANT_S = 0.10          # elevator servo time constant, s
THROTTLE_TIME_CONSTANT_S = 0.50          # engine/throttle time constant, s
MAX_ELEVATOR_RATE_RAD_S = np.deg2rad(60.0)  # max elevator slew rate, rad/s
MAX_THROTTLE_RATE_PER_S = 1.0            # max throttle slew rate, full range/s