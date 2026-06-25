# 6DOF Fixed-Wing Flight Dynamics and Control Simulation

## Motivation

I built this project to develop a working understanding of flight dynamics and control from the ground up. Rather than using an existing simulator, I implemented the physics, trim solver, and autopilot from scratch so I could understand every component before building on top of it.

## Overview

A modular fixed-wing aircraft flight simulation built in Python, covering longitudinal dynamics, full 6DOF rigid-body motion, trim analysis, and cascaded autopilot design.

The project was built in two stages:

1. **Longitudinal (2D) model** - translational and pitch dynamics, trim solver, and PID altitude/airspeed controllers
2. **Full 6DOF model** - extends to complete rigid-body dynamics in 3D, including roll, yaw, lateral forces, and a cascaded autopilot for altitude, airspeed, heading, and attitude hold

## Technical Details

**State vector (6DOF):**

```
[x, y, z, u, v, w, phi, theta, psi, p, q, r]
```

12 states: inertial position, body-axis velocities, Euler angles, body angular rates

**Core components:**
- Newton-Euler rigid-body equations of motion in body axes
- Linear stability derivative aerodynamic model (CL, CD, CM, CY, Cl, Cn)
- Direction Cosine Matrix (DCM) rotation between body and NED inertial frame
- Numerical trim solver using SciPy root-finding, solving for alpha, theta, elevator, and throttle at steady-level flight
- Cascaded PD/P autopilot with pitch and roll inner loops, altitude/airspeed/heading outer loops
- Actuator saturation and anti-windup
- Automated gain sweep with scoring across controller parameter combinations

## Project Structure

```
src/
├── sixdof/                          # Full 6DOF implementation
│   ├── dynamics_6dof.py             # Newton-Euler equations of motion
│   ├── aerodynamics_6dof.py         # Stability derivative aero model
│   ├── trim_6dof.py                 # Numerical trim solver
│   ├── controllers_6dof.py          # Cascaded PD/P autopilot
│   ├── rotations.py                 # DCM and kinematic equations
│   └── simulate_6dof_closed_loop.py # Closed-loop simulation runner
├── longitudinal_dynamics.py         # 2D longitudinal precursor model
├── trim_longitudinal.py             # Longitudinal trim solver
├── simulate_longitudinal.py         # Longitudinal simulation runner
├── controllers.py                   # Longitudinal PID controllers
├── aircraft_params.py               # Aircraft physical parameters
├── controller_config.py             # Controller gain settings
├── scenario_config.py               # Scenario definitions
├── run_scenarios.py                 # Batch scenario runner
├── run_gain_sweep.py                # Gain sweep and scoring
└── run_final_project.py             # Full validation runner
```

## Results

The closed-loop simulation was validated across multiple flight scenarios:
- Altitude step commands with and without wind gusts
- Airspeed step commands
- Heading step commands
- Combined altitude, airspeed, and heading maneuvers

Controller gains were selected via automated gain sweep, scoring each candidate on tracking error, settling time, and control effort.

## Dependencies

```
numpy
scipy
matplotlib
pandas
```

Install with:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root:

```bash
# Run longitudinal simulation
python src/simulate_longitudinal.py

# Run all scenarios
python src/run_scenarios.py

# Run gain sweep
python src/run_gain_sweep.py

# Run full 6DOF validation
python src/run_final_project.py
```
