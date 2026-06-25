# scenario_6dof_config.py
# Scenario configuration for the 6DOF fixed-wing simulation.
#
# This file defines simple open-loop test scenarios for the 6DOF aircraft.
#
# The first goal is not closed-loop control yet.
# The first goal is:
#   1. Trim the aircraft for straight and level flight.
#   2. Simulate the 12-state 6DOF equations.
#   3. Apply optional small control perturbations.
#   4. Plot the aircraft response.
#
# Coordinate convention:
#   Inertial frame:
#       +x = forward/north
#       +y = right/east
#       +z = down
#
#   Altitude is:
#       altitude = -z_down

from dataclasses import dataclass


@dataclass
class Scenario6DOFConfig:
    """
    Scenario configuration for open-loop 6DOF simulation.
    """

    # Scenario name for saved files
    scenario_name: str = "sixdof_trim_hold"

    # Trim / initial condition
    target_airspeed_mps: float = 25.0
    trim_altitude_m: float = 100.0

    # Simulation timing
    sim_start_time_s: float = 0.0
    sim_end_time_s: float = 30.0
    num_time_points: int = 2000

    # Optional open-loop control perturbation timing
    control_step_start_time_s: float = 5.0
    control_step_end_time_s: float = 7.0

    # Optional control perturbations added to trim controls
    elevator_step_deg: float = 0.0
    aileron_step_deg: float = 0.0
    rudder_step_deg: float = 0.0
    throttle_step: float = 0.0

    # Constant inertial wind in NED frame
    wind_x_inertial_mps: float = 0.0
    wind_y_inertial_mps: float = 0.0
    wind_z_down_inertial_mps: float = 0.0

    # Output settings
    save_plots: bool = True
    results_dir: str = "results/6dof"


DEFAULT_6DOF_SCENARIO = Scenario6DOFConfig()


ELEVATOR_PULSE_SCENARIO = Scenario6DOFConfig(
    scenario_name="sixdof_elevator_pulse",
    target_airspeed_mps=25.0,
    trim_altitude_m=100.0,
    sim_start_time_s=0.0,
    sim_end_time_s=30.0,
    num_time_points=2000,
    control_step_start_time_s=5.0,
    control_step_end_time_s=7.0,
    elevator_step_deg=-2.0,
    aileron_step_deg=0.0,
    rudder_step_deg=0.0,
    throttle_step=0.0,
)


AILERON_PULSE_SCENARIO = Scenario6DOFConfig(
    scenario_name="sixdof_aileron_pulse",
    target_airspeed_mps=25.0,
    trim_altitude_m=100.0,
    sim_start_time_s=0.0,
    sim_end_time_s=30.0,
    num_time_points=2000,
    control_step_start_time_s=5.0,
    control_step_end_time_s=7.0,
    elevator_step_deg=0.0,
    aileron_step_deg=5.0,
    rudder_step_deg=0.0,
    throttle_step=0.0,
)