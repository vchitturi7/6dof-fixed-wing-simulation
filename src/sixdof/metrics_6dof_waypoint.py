# metrics_6dof_waypoint.py
# Metrics for the final 6DOF waypoint-following simulation.
#
# Purpose:
#   Converts the final waypoint-following run into quantitative results.
#
# This supports the final project story:
#   - 6DOF dynamics implemented
#   - closed-loop autopilot implemented
#   - gain tuning validated
#   - waypoint guidance added
#   - final mission performance measured

from pathlib import Path
import csv

import numpy as np


def compute_waypoint_metrics(t, data, waypoints, scenario):
    """
    Computes final waypoint-following performance metrics.

    Inputs:
        t         = time array, s
        data      = waypoint-following post-processed data dictionary
        waypoints = list of Waypoint objects
        scenario  = WaypointFollowingScenario

    Returns:
        metrics dictionary
    """

    final_waypoint = waypoints[-1]

    final_x = data["x"][-1]
    final_y = data["y"][-1]
    final_altitude = data["altitude"][-1]
    final_airspeed = data["airspeed"][-1]

    final_horizontal_error_m = np.sqrt(
        (final_x - final_waypoint.x_m) ** 2
        + (final_y - final_waypoint.y_m) ** 2
    )

    final_altitude_error_m = final_waypoint.altitude_m - final_altitude
    final_airspeed_error_mps = final_waypoint.airspeed_mps - final_airspeed

    unique_waypoints_reached = sorted(
        set(
            int(index)
            for index in data["active_waypoint_index"]
        )
    )

    max_waypoint_index_reached = max(unique_waypoints_reached)

    reached_final_waypoint = (
        max_waypoint_index_reached >= len(waypoints) - 1
    )

    waypoint_switch_count = 0

    for i in range(1, len(data["active_waypoint_index"])):
        if data["active_waypoint_index"][i] != data["active_waypoint_index"][i - 1]:
            waypoint_switch_count += 1

    metrics = {
        # Scenario
        "scenario_name": scenario.scenario_name,
        "sim_end_time_s": scenario.sim_end_time_s,
        "time_step_s": scenario.time_step_s,
        "target_airspeed_mps": scenario.target_airspeed_mps,
        "trim_altitude_m": scenario.trim_altitude_m,

        # Waypoint configuration
        "number_of_waypoints": len(waypoints),
        "waypoint_capture_radius_m": scenario.waypoint_capture_radius_m,
        "max_heading_change_per_command_deg": scenario.max_heading_change_per_command_deg,

        # Final state
        "final_x_m": final_x,
        "final_y_m": final_y,
        "final_altitude_m": final_altitude,
        "final_airspeed_mps": final_airspeed,
        "final_heading_deg": np.rad2deg(data["psi"][-1]),

        # Final waypoint target
        "final_waypoint_x_m": final_waypoint.x_m,
        "final_waypoint_y_m": final_waypoint.y_m,
        "final_waypoint_altitude_m": final_waypoint.altitude_m,
        "final_waypoint_airspeed_mps": final_waypoint.airspeed_mps,

        # Final errors
        "final_horizontal_error_to_last_waypoint_m": final_horizontal_error_m,
        "final_altitude_error_m": final_altitude_error_m,
        "final_airspeed_error_mps": final_airspeed_error_mps,

        # Mission sequencing
        "max_waypoint_index_reached": max_waypoint_index_reached,
        "reached_final_waypoint": reached_final_waypoint,
        "waypoint_switch_count": waypoint_switch_count,

        # Path-following performance
        "max_abs_cross_track_error_m": np.max(
            np.abs(data["cross_track_error"])
        ),
        "mean_abs_cross_track_error_m": np.mean(
            np.abs(data["cross_track_error"])
        ),
        "final_distance_to_active_waypoint_m": data["distance_to_waypoint"][-1],

        # Flight envelope / control quality
        "max_abs_roll_deg": np.max(
            np.abs(np.rad2deg(data["phi"]))
        ),
        "max_abs_pitch_deg": np.max(
            np.abs(np.rad2deg(data["theta"]))
        ),
        "max_abs_heading_deg": np.max(
            np.abs(np.rad2deg(data["psi"]))
        ),
        "max_abs_alpha_deg": np.max(
            np.abs(np.rad2deg(data["alpha"]))
        ),
        "max_abs_beta_deg": np.max(
            np.abs(np.rad2deg(data["beta"]))
        ),
        "max_abs_p_deg_s": np.max(
            np.abs(np.rad2deg(data["p_rate"]))
        ),
        "max_abs_q_deg_s": np.max(
            np.abs(np.rad2deg(data["q_rate"]))
        ),
        "max_abs_r_deg_s": np.max(
            np.abs(np.rad2deg(data["r_rate"]))
        ),

        # Tracking relative to guidance commands
        "max_abs_altitude_command_error_m": np.max(
            np.abs(data["altitude_command"] - data["altitude"])
        ),
        "final_altitude_command_error_m": (
            data["altitude_command"][-1] - data["altitude"][-1]
        ),
        "max_abs_airspeed_command_error_mps": np.max(
            np.abs(data["airspeed_command"] - data["airspeed"])
        ),
        "final_airspeed_command_error_mps": (
            data["airspeed_command"][-1] - data["airspeed"][-1]
        ),
    }

    return metrics


def print_waypoint_metrics(metrics):
    """
    Prints waypoint-following metrics.
    """

    print("\nWaypoint-following metrics:")
    print("---------------------------")
    print(f"Scenario:                              {metrics['scenario_name']}")
    print(f"Reached final waypoint:                {metrics['reached_final_waypoint']}")
    print(f"Max waypoint index reached:            {metrics['max_waypoint_index_reached']}")
    print(f"Waypoint switches:                     {metrics['waypoint_switch_count']}")
    print(f"Final x position:                      {metrics['final_x_m']:.3f} m")
    print(f"Final y position:                      {metrics['final_y_m']:.3f} m")
    print(f"Final altitude:                        {metrics['final_altitude_m']:.3f} m")
    print(f"Final airspeed:                        {metrics['final_airspeed_mps']:.3f} m/s")
    print(f"Final horizontal error to last WP:      {metrics['final_horizontal_error_to_last_waypoint_m']:.3f} m")
    print(f"Final altitude error to last WP:        {metrics['final_altitude_error_m']:.3f} m")
    print(f"Final airspeed error to last WP:        {metrics['final_airspeed_error_mps']:.3f} m/s")
    print(f"Max abs cross-track error:             {metrics['max_abs_cross_track_error_m']:.3f} m")
    print(f"Mean abs cross-track error:            {metrics['mean_abs_cross_track_error_m']:.3f} m")
    print(f"Max abs roll:                          {metrics['max_abs_roll_deg']:.3f} deg")
    print(f"Max abs pitch:                         {metrics['max_abs_pitch_deg']:.3f} deg")
    print(f"Max abs sideslip:                      {metrics['max_abs_beta_deg']:.3f} deg")
    print(f"Max altitude command error:            {metrics['max_abs_altitude_command_error_m']:.3f} m")
    print(f"Max airspeed command error:            {metrics['max_abs_airspeed_command_error_mps']:.3f} m/s")


def save_waypoint_metrics_to_csv(metrics, output_path):
    """
    Saves waypoint metrics to a one-row CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(metrics.keys()),
        )

        writer.writeheader()
        writer.writerow(metrics)

    print(f"Saved waypoint metrics CSV: {output_path}")