# metrics_6dof_closed_loop.py
# Metrics for closed-loop 6DOF autopilot simulations.
#
# These metrics quantify how well the 6DOF aircraft tracks:
#   - altitude command
#   - airspeed command
#   - heading command
#   - roll command
#   - pitch command
#
# They also measure:
#   - control effort
#   - max body rates
#   - max sideslip
#   - final tracking errors
#
# This is the closed-loop 6DOF equivalent of the longitudinal metrics.

import numpy as np

from rotations import wrap_angle_pi


def compute_settling_time(t, error, tolerance, start_time):
    """
    Computes approximate settling time.

    Settling time is defined as the first time after start_time
    when abs(error) stays within tolerance for the rest of the simulation.

    Returns:
        settling_time_s, or None if the signal never settles.
    """

    start_index = np.searchsorted(t, start_time)

    for i in range(start_index, len(t)):
        if np.all(np.abs(error[i:]) <= tolerance):
            return t[i] - start_time

    return None


def compute_heading_error_array(heading_command, heading_actual):
    """
    Computes wrapped heading error array in radians.

    Uses wrap_angle_pi so heading error is always the shortest signed error.
    """

    heading_error = []

    for command_i, actual_i in zip(heading_command, heading_actual):
        heading_error.append(
            wrap_angle_pi(command_i - actual_i)
        )

    return np.array(heading_error)


def compute_closed_loop_6dof_metrics(t, data, config):
    """
    Computes closed-loop 6DOF tracking metrics.

    Inputs:
        t      = time array, s
        data   = post-processed data dictionary from simulate_6dof_closed_loop.py
        config = ClosedLoop6DOFScenario

    Returns:
        metrics dictionary
    """

    altitude_error = data["altitude_command"] - data["altitude"]
    airspeed_error = data["airspeed_command"] - data["airspeed"]
    heading_error = compute_heading_error_array(
        heading_command=data["heading_command"],
        heading_actual=data["psi"],
    )

    roll_error = data["phi_command"] - data["phi"]
    pitch_error = data["theta_command"] - data["theta"]

    after_command = t >= config.command_start_time_s

    altitude_settling_time_s = compute_settling_time(
        t=t,
        error=altitude_error,
        tolerance=1.0,
        start_time=config.command_start_time_s,
    )

    airspeed_settling_time_s = compute_settling_time(
        t=t,
        error=airspeed_error,
        tolerance=0.5,
        start_time=config.command_start_time_s,
    )

    heading_settling_time_s = compute_settling_time(
        t=t,
        error=np.rad2deg(heading_error),
        tolerance=2.0,
        start_time=config.command_start_time_s,
    )

    metrics = {
        # Scenario setup
        "scenario_name": config.scenario_name,
        "target_airspeed_mps": config.target_airspeed_mps,
        "trim_altitude_m": config.trim_altitude_m,
        "altitude_step_m": config.altitude_step_m,
        "airspeed_step_mps": config.airspeed_step_mps,
        "heading_step_deg": config.heading_step_deg,
        "sim_end_time_s": config.sim_end_time_s,

        # Final commanded values
        "final_altitude_command_m": data["altitude_command"][-1],
        "final_airspeed_command_mps": data["airspeed_command"][-1],
        "final_heading_command_deg": np.rad2deg(data["heading_command"][-1]),

        # Final actual values
        "final_x_m": data["x"][-1],
        "final_y_m": data["y"][-1],
        "final_altitude_m": data["altitude"][-1],
        "final_airspeed_mps": data["airspeed"][-1],
        "final_heading_deg": np.rad2deg(data["psi"][-1]),
        "final_roll_deg": np.rad2deg(data["phi"][-1]),
        "final_pitch_deg": np.rad2deg(data["theta"][-1]),

        # Final errors
        "final_altitude_error_m": altitude_error[-1],
        "final_airspeed_error_mps": airspeed_error[-1],
        "final_heading_error_deg": np.rad2deg(heading_error[-1]),
        "final_roll_error_deg": np.rad2deg(roll_error[-1]),
        "final_pitch_error_deg": np.rad2deg(pitch_error[-1]),

        # Max tracking errors after command starts
        "max_abs_altitude_error_after_command_m": np.max(
            np.abs(altitude_error[after_command])
        ),
        "max_abs_airspeed_error_after_command_mps": np.max(
            np.abs(airspeed_error[after_command])
        ),
        "max_abs_heading_error_after_command_deg": np.max(
            np.abs(np.rad2deg(heading_error[after_command]))
        ),
        "max_abs_roll_error_after_command_deg": np.max(
            np.abs(np.rad2deg(roll_error[after_command]))
        ),
        "max_abs_pitch_error_after_command_deg": np.max(
            np.abs(np.rad2deg(pitch_error[after_command]))
        ),

        # Settling times
        "altitude_settling_time_s": altitude_settling_time_s,
        "airspeed_settling_time_s": airspeed_settling_time_s,
        "heading_settling_time_s": heading_settling_time_s,

        # Attitude and rate limits
        "max_abs_roll_deg": np.max(np.abs(np.rad2deg(data["phi"]))),
        "max_abs_pitch_deg": np.max(np.abs(np.rad2deg(data["theta"]))),
        "max_abs_heading_deg": np.max(np.abs(np.rad2deg(data["psi"]))),
        "max_abs_alpha_deg": np.max(np.abs(np.rad2deg(data["alpha"]))),
        "max_abs_beta_deg": np.max(np.abs(np.rad2deg(data["beta"]))),
        "max_abs_p_deg_s": np.max(np.abs(np.rad2deg(data["p_rate"]))),
        "max_abs_q_deg_s": np.max(np.abs(np.rad2deg(data["q_rate"]))),
        "max_abs_r_deg_s": np.max(np.abs(np.rad2deg(data["r_rate"]))),

        # Control effort
        "max_abs_elevator_deg": np.max(np.abs(np.rad2deg(data["delta_e"]))),
        "max_abs_aileron_deg": np.max(np.abs(np.rad2deg(data["delta_a"]))),
        "max_abs_rudder_deg": np.max(np.abs(np.rad2deg(data["delta_r"]))),
        "max_throttle": np.max(data["delta_t"]),
        "min_throttle": np.min(data["delta_t"]),
    }

    return metrics


def print_closed_loop_6dof_metrics(metrics):
    """
    Prints readable closed-loop 6DOF metrics.
    """

    print("\nClosed-loop 6DOF metrics:")
    print("-------------------------")
    print(f"Scenario:                             {metrics['scenario_name']}")
    print(f"Final altitude:                       {metrics['final_altitude_m']:.3f} m")
    print(f"Final airspeed:                       {metrics['final_airspeed_mps']:.3f} m/s")
    print(f"Final heading:                        {metrics['final_heading_deg']:.3f} deg")
    print(f"Final altitude error:                 {metrics['final_altitude_error_m']:.3f} m")
    print(f"Final airspeed error:                 {metrics['final_airspeed_error_mps']:.3f} m/s")
    print(f"Final heading error:                  {metrics['final_heading_error_deg']:.3f} deg")
    print(f"Max altitude error after command:     {metrics['max_abs_altitude_error_after_command_m']:.3f} m")
    print(f"Max airspeed error after command:     {metrics['max_abs_airspeed_error_after_command_mps']:.3f} m/s")
    print(f"Max heading error after command:      {metrics['max_abs_heading_error_after_command_deg']:.3f} deg")
    print(f"Max roll error after command:         {metrics['max_abs_roll_error_after_command_deg']:.3f} deg")
    print(f"Max pitch error after command:        {metrics['max_abs_pitch_error_after_command_deg']:.3f} deg")
    print(f"Max abs roll angle:                   {metrics['max_abs_roll_deg']:.3f} deg")
    print(f"Max abs pitch angle:                  {metrics['max_abs_pitch_deg']:.3f} deg")
    print(f"Max abs sideslip:                     {metrics['max_abs_beta_deg']:.3f} deg")
    print(f"Max abs elevator:                     {metrics['max_abs_elevator_deg']:.3f} deg")
    print(f"Max abs aileron:                      {metrics['max_abs_aileron_deg']:.3f} deg")
    print(f"Max abs rudder:                       {metrics['max_abs_rudder_deg']:.3f} deg")
    print(f"Throttle range:                       {metrics['min_throttle']:.3f} to {metrics['max_throttle']:.3f}")

    if metrics["altitude_settling_time_s"] is None:
        print("Altitude settling time ±1 m:          Did not settle")
    else:
        print(f"Altitude settling time ±1 m:          {metrics['altitude_settling_time_s']:.3f} s")

    if metrics["airspeed_settling_time_s"] is None:
        print("Airspeed settling time ±0.5 m/s:      Did not settle")
    else:
        print(f"Airspeed settling time ±0.5 m/s:      {metrics['airspeed_settling_time_s']:.3f} s")

    if metrics["heading_settling_time_s"] is None:
        print("Heading settling time ±2 deg:         Did not settle")
    else:
        print(f"Heading settling time ±2 deg:         {metrics['heading_settling_time_s']:.3f} s")