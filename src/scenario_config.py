# scenario_config.py
# Central place to define simulation scenario settings.
#
# This lets you change altitude commands, airspeed commands,
# gust timing, gust strength, simulation duration, and output settings
# without editing the main simulation logic.

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """
    Scenario configuration for the longitudinal GNC simulation.

    Units:
        target_airspeed_mps: m/s
        trim_altitude_m: m
        altitude_step_m: m
        command_start_time_s: s
        sim_start_time_s: s
        sim_end_time_s: s
        num_time_points: unitless
        gust_start_time_s: s
        gust_end_time_s: s
        gust_wind_x_mps: m/s
        gust_wind_z_mps: m/s
    """

    # Scenario name for saved files
    scenario_name: str = "altitude_airspeed_hold_smooth_gust"

    # Trim / initial flight condition
    target_airspeed_mps: float = 25.0
    trim_altitude_m: float = 100.0

    # Command scenario
    command_start_time_s: float = 5.0
    altitude_step_m: float = 20.0

    # Simulation time settings
    sim_start_time_s: float = 0.0
    sim_end_time_s: float = 70.0
    num_time_points: int = 3000

    # Gust disturbance settings
    gust_start_time_s: float = 25.0
    gust_end_time_s: float = 30.0

    # Inertial-frame wind convention:
    # +x = forward/horizontal
    # +z = upward
    gust_wind_x_mps: float = 0.0
    gust_wind_z_mps: float = 2.0

    # If True, use smooth half-cosine gust ramp instead of square gust
    use_smooth_gust: bool = True

    # Performance metric tolerances
    altitude_settling_tolerance_m: float = 1.0
    airspeed_settling_tolerance_mps: float = 0.25

    # Output settings
    save_plots: bool = True
    results_dir: str = "results"


# Default scenario used by simulate_longitudinal.py
DEFAULT_SCENARIO = ScenarioConfig()