# run_logging.py
# Utility functions for saving simulation configuration and run metadata.
#
# Purpose:
#   For every simulation run, save the exact scenario settings,
#   controller gains, trim solution, aircraft parameters, actuator settings,
#   and basic metadata to JSON.
#
# This makes plots and metrics reproducible.

from pathlib import Path
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math

import aircraft_params as p


def dataclass_to_dict(obj):
    """
    Converts a dataclass object to a dictionary.

    Inputs:
        obj = dataclass instance

    Returns:
        dictionary representation of dataclass
    """

    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass object, got {type(obj)}")

    return asdict(obj)


def make_json_safe(value):
    """
    Converts NumPy/Python values into JSON-safe values.

    This handles:
        - NumPy scalar values
        - NaN / inf values
        - nested dictionaries
        - lists / tuples
    """

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Handles NumPy scalar values without directly importing NumPy here.
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return str(value)

    return value


def get_aircraft_and_actuator_settings():
    """
    Returns aircraft, aerodynamic, propulsion, and actuator settings.

    These values strongly affect simulation behavior, so saving them
    makes each run easier to reproduce later.
    """

    return {
        "physical_constants": {
            "gravity_mps2": p.G,
            "air_density_kgpm3": p.RHO,
        },
        "aircraft_properties": {
            "mass_kg": p.MASS,
            "iyy_kg_m2": p.IYY,
            "wing_area_m2": p.S,
            "mean_aerodynamic_chord_m": p.C,
        },
        "aerodynamic_coefficients": {
            "CL0": p.CL0,
            "CL_ALPHA": p.CL_ALPHA,
            "CL_Q": p.CL_Q,
            "CL_DE": p.CL_DE,
            "CD0": p.CD0,
            "CD_ALPHA": p.CD_ALPHA,
            "CD_DE": p.CD_DE,
            "CM0": p.CM0,
            "CM_ALPHA": p.CM_ALPHA,
            "CM_Q": p.CM_Q,
            "CM_DE": p.CM_DE,
        },
        "propulsion": {
            "max_thrust_N": p.MAX_THRUST,
        },
        "actuator_dynamics": {
            "elevator_time_constant_s": p.ELEVATOR_TIME_CONSTANT_S,
            "throttle_time_constant_s": p.THROTTLE_TIME_CONSTANT_S,
            "max_elevator_rate_rad_s": p.MAX_ELEVATOR_RATE_RAD_S,
            "max_elevator_rate_deg_s": math.degrees(p.MAX_ELEVATOR_RATE_RAD_S),
            "max_throttle_rate_per_s": p.MAX_THROTTLE_RATE_PER_S,
        },
    }


def save_run_config(
    config,
    gains,
    trim_info,
    output_dir,
    filename_prefix,
    extra_metadata=None,
):
    """
    Saves a JSON file containing the full run configuration.

    Inputs:
        config          = ScenarioConfig dataclass
        gains           = ControllerGains dataclass
        trim_info       = dictionary returned by solve_trim()
        output_dir      = folder where JSON should be saved
        filename_prefix = prefix for JSON filename
        extra_metadata  = optional extra dictionary

    Returns:
        output_path = path to saved JSON file
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    run_metadata = {
        "created_at_utc": created_at_utc,
        "filename_prefix": filename_prefix,
        "description": (
            "Longitudinal fixed-wing GNC simulation run configuration. "
            "Includes scenario settings, controller gains, trim solution, "
            "aircraft parameters, aerodynamic coefficients, propulsion settings, "
            "and actuator dynamics."
        ),
    }

    if extra_metadata is not None:
        run_metadata["extra_metadata"] = extra_metadata

    run_data = {
        "metadata": run_metadata,
        "scenario_config": dataclass_to_dict(config),
        "controller_gains": dataclass_to_dict(gains),
        "trim_solution": trim_info,
        "aircraft_and_actuator_settings": get_aircraft_and_actuator_settings(),
    }

    run_data = make_json_safe(run_data)

    output_path = output_dir / f"{filename_prefix}_run_config.json"

    with output_path.open(mode="w", encoding="utf-8") as json_file:
        json.dump(run_data, json_file, indent=4)

    print(f"Saved run config JSON: {output_path}")

    return output_path