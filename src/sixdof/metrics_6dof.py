# metrics_6dof.py
# Performance and sanity metrics for 6DOF open-loop simulations.
#
# These metrics are intentionally simple for now.
# The goal is to quantify whether the 6DOF aircraft response is reasonable
# across trim-hold and open-loop perturbation scenarios.
#
# Later, after closed-loop control is added, this file will expand to include:
#   - altitude tracking error
#   - airspeed tracking error
#   - heading tracking error
#   - waypoint cross-track error
#   - settling times
#   - actuator saturation percentages

import numpy as np


def compute_6dof_metrics(t, data, config, trim_info):
    """
    Computes metrics for an open-loop 6DOF simulation.

    Inputs:
        t         = time array, s
        data      = post-processed data dictionary from simulate_6dof.py
        config    = Scenario6DOFConfig
        trim_info = trim information dictionary from solve_trim_6dof()

    Returns:
        metrics dictionary
    """

    altitude = data["altitude"]
    airspeed = data["airspeed"]

    phi = data["phi"]
    theta = data["theta"]
    psi = data["psi"]

    alpha = data["alpha"]
    beta = data["beta"]

    p_rate = data["p_rate"]
    q_rate = data["q_rate"]
    r_rate = data["r_rate"]

    delta_e = data["delta_e"]
    delta_a = data["delta_a"]
    delta_r = data["delta_r"]
    delta_t = data["delta_t"]

    initial_altitude = altitude[0]
    initial_airspeed = airspeed[0]

    altitude_deviation = altitude - initial_altitude
    airspeed_deviation = airspeed - initial_airspeed

    metrics = {
        # Scenario information
        "scenario_name": config.scenario_name,
        "target_airspeed_mps": config.target_airspeed_mps,
        "trim_altitude_m": config.trim_altitude_m,
        "sim_end_time_s": config.sim_end_time_s,
        "elevator_step_deg": config.elevator_step_deg,
        "aileron_step_deg": config.aileron_step_deg,
        "rudder_step_deg": config.rudder_step_deg,
        "throttle_step": config.throttle_step,

        # Trim solution
        "alpha_trim_deg": trim_info["alpha_trim_deg"],
        "theta_trim_deg": trim_info["theta_trim_deg"],
        "delta_e_trim_deg": trim_info["delta_e_trim_deg"],
        "delta_t_trim": trim_info["delta_t_trim"],

        # Final state summary
        "final_x_m": data["x"][-1],
        "final_y_m": data["y"][-1],
        "final_altitude_m": altitude[-1],
        "final_airspeed_mps": airspeed[-1],
        "final_roll_deg": np.rad2deg(phi[-1]),
        "final_pitch_deg": np.rad2deg(theta[-1]),
        "final_heading_deg": np.rad2deg(psi[-1]),
        "final_alpha_deg": np.rad2deg(alpha[-1]),
        "final_beta_deg": np.rad2deg(beta[-1]),

        # Open-loop deviation metrics
        "max_abs_altitude_deviation_m": np.max(np.abs(altitude_deviation)),
        "final_altitude_deviation_m": altitude_deviation[-1],
        "max_abs_airspeed_deviation_mps": np.max(np.abs(airspeed_deviation)),
        "final_airspeed_deviation_mps": airspeed_deviation[-1],

        # Attitude/rate metrics
        "max_abs_roll_deg": np.max(np.abs(np.rad2deg(phi))),
        "max_abs_pitch_deg": np.max(np.abs(np.rad2deg(theta))),
        "max_abs_heading_deg": np.max(np.abs(np.rad2deg(psi))),
        "max_abs_alpha_deg": np.max(np.abs(np.rad2deg(alpha))),
        "max_abs_beta_deg": np.max(np.abs(np.rad2deg(beta))),
        "max_abs_p_deg_s": np.max(np.abs(np.rad2deg(p_rate))),
        "max_abs_q_deg_s": np.max(np.abs(np.rad2deg(q_rate))),
        "max_abs_r_deg_s": np.max(np.abs(np.rad2deg(r_rate))),

        # Control usage metrics
        "max_abs_elevator_deg": np.max(np.abs(np.rad2deg(delta_e))),
        "max_abs_aileron_deg": np.max(np.abs(np.rad2deg(delta_a))),
        "max_abs_rudder_deg": np.max(np.abs(np.rad2deg(delta_r))),
        "max_throttle": np.max(delta_t),
        "min_throttle": np.min(delta_t),
    }

    return metrics


def print_6dof_metrics(metrics):
    """
    Prints a readable summary of 6DOF metrics.
    """

    print("\n6DOF metrics:")
    print("-------------")
    print(f"Scenario:                         {metrics['scenario_name']}")
    print(f"Final x position:                 {metrics['final_x_m']:.3f} m")
    print(f"Final y position:                 {metrics['final_y_m']:.3f} m")
    print(f"Final altitude:                   {metrics['final_altitude_m']:.3f} m")
    print(f"Final airspeed:                   {metrics['final_airspeed_mps']:.3f} m/s")
    print(f"Max altitude deviation:           {metrics['max_abs_altitude_deviation_m']:.6f} m")
    print(f"Max airspeed deviation:           {metrics['max_abs_airspeed_deviation_mps']:.6f} m/s")
    print(f"Max abs roll angle:               {metrics['max_abs_roll_deg']:.3f} deg")
    print(f"Max abs pitch angle:              {metrics['max_abs_pitch_deg']:.3f} deg")
    print(f"Max abs heading angle:            {metrics['max_abs_heading_deg']:.3f} deg")
    print(f"Max abs AoA:                      {metrics['max_abs_alpha_deg']:.3f} deg")
    print(f"Max abs sideslip:                 {metrics['max_abs_beta_deg']:.3f} deg")
    print(f"Max abs roll rate p:              {metrics['max_abs_p_deg_s']:.3f} deg/s")
    print(f"Max abs pitch rate q:             {metrics['max_abs_q_deg_s']:.3f} deg/s")
    print(f"Max abs yaw rate r:               {metrics['max_abs_r_deg_s']:.3f} deg/s")
    print(f"Max abs elevator:                 {metrics['max_abs_elevator_deg']:.3f} deg")
    print(f"Max abs aileron:                  {metrics['max_abs_aileron_deg']:.3f} deg")
    print(f"Max abs rudder:                   {metrics['max_abs_rudder_deg']:.3f} deg")
    print(f"Throttle range:                   {metrics['min_throttle']:.3f} to {metrics['max_throttle']:.3f}")