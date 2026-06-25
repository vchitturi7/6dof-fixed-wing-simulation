# run_6dof_gain_comparison.py
# Compares multiple 6DOF closed-loop controller gain sets.
#
# Usage from project root:
#   python src/sixdof/run_6dof_gain_comparison.py
#
# Purpose:
#   This script makes controller tuning more professional.
#   Instead of saying "I changed gains until it looked good," this lets you say:
#
#   "I evaluated multiple gain sets across standardized 6DOF scenarios using
#    tracking error, settling time, sideslip, body-rate, and actuator-usage metrics."
#
# It runs:
#   - baseline gains
#   - lateral-tuned gains
#   - longitudinal+tuned gains
#   - aggressive longitudinal gains
#
# Across:
#   - altitude step
#   - heading step
#   - altitude + heading step
#   - airspeed step
#   - large combined maneuver
#
# It saves:
#   results/6dof/gain_comparison/gain_comparison_metrics.csv
#   results/6dof/gain_comparison/gain_comparison_ranked.csv

from pathlib import Path
import csv
import copy

import numpy as np

from controllers_6dof import Controller6DOFGains
from simulate_6dof_closed_loop import (
    ClosedLoop6DOFScenario,
    run_closed_loop_6dof,
)
from metrics_6dof_closed_loop import compute_closed_loop_6dof_metrics


def make_scenario(
    scenario_name,
    altitude_step_m,
    heading_step_deg,
    airspeed_step_mps,
    sim_end_time_s=60.0,
):
    """
    Creates one closed-loop 6DOF scenario.
    """

    scenario = ClosedLoop6DOFScenario()

    scenario.scenario_name = scenario_name
    scenario.altitude_step_m = altitude_step_m
    scenario.heading_step_deg = heading_step_deg
    scenario.airspeed_step_mps = airspeed_step_mps
    scenario.sim_end_time_s = sim_end_time_s
    scenario.save_plots = False
    scenario.results_dir = "results/6dof/gain_comparison"

    return scenario


def make_gain_sets():
    """
    Creates gain sets to compare.

    Gain set 1:
        original_baseline
        The first closed-loop gain set.

    Gain set 2:
        lateral_tuned
        Reduced lateral aggressiveness and rudder saturation.

    Gain set 3:
        final_tuned
        Current preferred gain set after lateral and longitudinal tuning.

    Gain set 4:
        aggressive_longitudinal
        Tests whether stronger pitch/altitude gains reduce altitude drift
        during airspeed steps, at the cost of possible elevator/pitch effort.
    """

    original_baseline = Controller6DOFGains(
        pitch_kp=1.0,
        pitch_kd=0.35,
        altitude_kh_rad_per_m=0.0035,
        max_pitch_command_offset_rad=np.deg2rad(10.0),
        airspeed_kv=0.12,
        roll_kp=1.8,
        roll_kd=0.45,
        heading_kp_rad_per_rad=0.8,
        max_roll_command_rad=np.deg2rad(30.0),
        yaw_damper_kr=0.25,
        sideslip_kb=0.10,
    )

    lateral_tuned = Controller6DOFGains(
        pitch_kp=1.0,
        pitch_kd=0.40,
        altitude_kh_rad_per_m=0.0050,
        max_pitch_command_offset_rad=np.deg2rad(10.0),
        airspeed_kv=0.09,
        roll_kp=0.85,
        roll_kd=0.90,
        heading_kp_rad_per_rad=0.35,
        max_roll_command_rad=np.deg2rad(18.0),
        yaw_damper_kr=0.08,
        sideslip_kb=0.03,
    )

    final_tuned = Controller6DOFGains(
        pitch_kp=1.25,
        pitch_kd=0.50,
        altitude_kh_rad_per_m=0.0065,
        max_pitch_command_offset_rad=np.deg2rad(12.0),
        airspeed_kv=0.075,
        roll_kp=0.85,
        roll_kd=0.90,
        heading_kp_rad_per_rad=0.35,
        max_roll_command_rad=np.deg2rad(18.0),
        yaw_damper_kr=0.08,
        sideslip_kb=0.03,
    )

    aggressive_longitudinal = Controller6DOFGains(
        pitch_kp=1.45,
        pitch_kd=0.60,
        altitude_kh_rad_per_m=0.0080,
        max_pitch_command_offset_rad=np.deg2rad(14.0),
        airspeed_kv=0.065,
        roll_kp=0.85,
        roll_kd=0.90,
        heading_kp_rad_per_rad=0.35,
        max_roll_command_rad=np.deg2rad(18.0),
        yaw_damper_kr=0.08,
        sideslip_kb=0.03,
    )

    return {
        "original_baseline": original_baseline,
        "lateral_tuned": lateral_tuned,
        "final_tuned": final_tuned,
        "aggressive_longitudinal": aggressive_longitudinal,
    }


def make_scenarios():
    """
    Creates standardized closed-loop validation scenarios.
    """

    return [
        make_scenario(
            scenario_name="altitude_step",
            altitude_step_m=20.0,
            heading_step_deg=0.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="heading_step",
            altitude_step_m=0.0,
            heading_step_deg=20.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="altitude_heading_step",
            altitude_step_m=20.0,
            heading_step_deg=20.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=70.0,
        ),
        make_scenario(
            scenario_name="airspeed_step",
            altitude_step_m=0.0,
            heading_step_deg=0.0,
            airspeed_step_mps=3.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="large_combined_step",
            altitude_step_m=40.0,
            heading_step_deg=35.0,
            airspeed_step_mps=3.0,
            sim_end_time_s=90.0,
        ),
    ]


def safe_metric_value(value, penalty_value):
    """
    Converts None metrics into finite penalty values for scoring.
    """

    if value is None:
        return penalty_value

    return value


def compute_gain_score(metrics):
    """
    Computes a single scalar score for ranking gain sets.

    Lower score is better.

    This score intentionally balances:
        tracking accuracy
        settling time
        lateral stability
        actuator usage

    It is not a universal flight-control metric. It is a practical engineering
    score for comparing candidate gain sets in this simulation.
    """

    altitude_error_score = 1.0 * metrics["max_abs_altitude_error_after_command_m"]
    airspeed_error_score = 8.0 * metrics["max_abs_airspeed_error_after_command_mps"]
    heading_error_score = 1.5 * metrics["max_abs_heading_error_after_command_deg"]

    roll_angle_score = 0.4 * metrics["max_abs_roll_deg"]
    pitch_angle_score = 0.4 * metrics["max_abs_pitch_deg"]
    sideslip_score = 3.0 * metrics["max_abs_beta_deg"]

    body_rate_score = (
        0.15 * metrics["max_abs_p_deg_s"]
        + 0.15 * metrics["max_abs_q_deg_s"]
        + 0.15 * metrics["max_abs_r_deg_s"]
    )

    actuator_score = (
        0.2 * metrics["max_abs_elevator_deg"]
        + 0.2 * metrics["max_abs_aileron_deg"]
        + 0.2 * metrics["max_abs_rudder_deg"]
    )

    throttle_score = (
        3.0 * abs(metrics["max_throttle"] - 0.5)
        + 3.0 * abs(metrics["min_throttle"] - 0.5)
    )

    altitude_settling = safe_metric_value(
        metrics["altitude_settling_time_s"],
        penalty_value=60.0,
    )

    airspeed_settling = safe_metric_value(
        metrics["airspeed_settling_time_s"],
        penalty_value=60.0,
    )

    heading_settling = safe_metric_value(
        metrics["heading_settling_time_s"],
        penalty_value=60.0,
    )

    settling_score = (
        0.15 * altitude_settling
        + 0.15 * airspeed_settling
        + 0.15 * heading_settling
    )

    total_score = (
        altitude_error_score
        + airspeed_error_score
        + heading_error_score
        + roll_angle_score
        + pitch_angle_score
        + sideslip_score
        + body_rate_score
        + actuator_score
        + throttle_score
        + settling_score
    )

    return total_score


def save_metrics_to_csv(metrics_list, output_path):
    """
    Saves list of metrics dictionaries to CSV.
    """

    if not metrics_list:
        print("No metrics to save.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics_list[0].keys())

    with output_path.open(mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            writer.writerow(metrics)

    print(f"Saved CSV: {output_path}")


def print_gain_summary(ranked_metrics):
    """
    Prints ranked gain comparison summary.
    """

    print("\nGain comparison summary:")
    print("------------------------")

    for i, metrics in enumerate(ranked_metrics, start=1):
        print(
            f"{i}. {metrics['gain_set']} | "
            f"{metrics['scenario_name']} | "
            f"score = {metrics['score']:.3f} | "
            f"final alt err = {metrics['final_altitude_error_m']:.3f} m | "
            f"final V err = {metrics['final_airspeed_error_mps']:.3f} m/s | "
            f"final hdg err = {metrics['final_heading_error_deg']:.3f} deg | "
            f"max ail = {metrics['max_abs_aileron_deg']:.2f} deg | "
            f"max rud = {metrics['max_abs_rudder_deg']:.2f} deg | "
            f"max beta = {metrics['max_abs_beta_deg']:.2f} deg"
        )


def summarize_by_gain_set(all_metrics):
    """
    Creates one aggregate row per gain set.
    """

    gain_sets = sorted(set(row["gain_set"] for row in all_metrics))

    summary_rows = []

    for gain_set in gain_sets:
        rows = [
            row
            for row in all_metrics
            if row["gain_set"] == gain_set
        ]

        scores = np.array([row["score"] for row in rows])

        final_altitude_errors = np.array([
            abs(row["final_altitude_error_m"])
            for row in rows
        ])

        final_airspeed_errors = np.array([
            abs(row["final_airspeed_error_mps"])
            for row in rows
        ])

        final_heading_errors = np.array([
            abs(row["final_heading_error_deg"])
            for row in rows
        ])

        max_ailerons = np.array([
            row["max_abs_aileron_deg"]
            for row in rows
        ])

        max_rudders = np.array([
            row["max_abs_rudder_deg"]
            for row in rows
        ])

        max_sideslips = np.array([
            row["max_abs_beta_deg"]
            for row in rows
        ])

        summary_row = {
            "gain_set": gain_set,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "mean_abs_final_altitude_error_m": float(np.mean(final_altitude_errors)),
            "mean_abs_final_airspeed_error_mps": float(np.mean(final_airspeed_errors)),
            "mean_abs_final_heading_error_deg": float(np.mean(final_heading_errors)),
            "max_abs_aileron_deg_across_scenarios": float(np.max(max_ailerons)),
            "max_abs_rudder_deg_across_scenarios": float(np.max(max_rudders)),
            "max_abs_beta_deg_across_scenarios": float(np.max(max_sideslips)),
        }

        summary_rows.append(summary_row)

    summary_rows = sorted(
        summary_rows,
        key=lambda row: row["mean_score"],
    )

    return summary_rows


def print_aggregate_summary(summary_rows):
    """
    Prints aggregate gain-set ranking.
    """

    print("\nAggregate gain-set ranking:")
    print("---------------------------")

    for i, row in enumerate(summary_rows, start=1):
        print(
            f"{i}. {row['gain_set']} | "
            f"mean score = {row['mean_score']:.3f} | "
            f"max score = {row['max_score']:.3f} | "
            f"mean abs alt err = {row['mean_abs_final_altitude_error_m']:.3f} m | "
            f"mean abs V err = {row['mean_abs_final_airspeed_error_mps']:.3f} m/s | "
            f"mean abs hdg err = {row['mean_abs_final_heading_error_deg']:.3f} deg | "
            f"max ail = {row['max_abs_aileron_deg_across_scenarios']:.2f} deg | "
            f"max rud = {row['max_abs_rudder_deg_across_scenarios']:.2f} deg | "
            f"max beta = {row['max_abs_beta_deg_across_scenarios']:.2f} deg"
        )


def main():
    """
    Runs gain comparison campaign.
    """

    output_dir = Path("results/6dof/gain_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    gain_sets = make_gain_sets()
    scenarios = make_scenarios()

    all_metrics = []

    for gain_set_name, gains in gain_sets.items():
        print("\n" + "#" * 90)
        print(f"Testing gain set: {gain_set_name}")
        print("#" * 90)

        for base_scenario in scenarios:
            scenario = copy.deepcopy(base_scenario)
            scenario.scenario_name = f"{gain_set_name}_{base_scenario.scenario_name}"

            print("\n" + "=" * 80)
            print(f"Running scenario: {scenario.scenario_name}")
            print("=" * 80)

            t, sol, data = run_closed_loop_6dof(
                config=scenario,
                gains=gains,
                show_plots=False,
            )

            metrics = compute_closed_loop_6dof_metrics(
                t=t,
                data=data,
                config=scenario,
            )

            metrics["gain_set"] = gain_set_name
            metrics["base_scenario"] = base_scenario.scenario_name
            metrics["score"] = compute_gain_score(metrics)

            all_metrics.append(metrics)

    ranked_metrics = sorted(
        all_metrics,
        key=lambda row: row["score"],
    )

    summary_rows = summarize_by_gain_set(all_metrics)

    save_metrics_to_csv(
        metrics_list=all_metrics,
        output_path=output_dir / "gain_comparison_metrics.csv",
    )

    save_metrics_to_csv(
        metrics_list=ranked_metrics,
        output_path=output_dir / "gain_comparison_ranked.csv",
    )

    save_metrics_to_csv(
        metrics_list=summary_rows,
        output_path=output_dir / "gain_set_summary.csv",
    )

    print_gain_summary(ranked_metrics)
    print_aggregate_summary(summary_rows)


if __name__ == "__main__":
    main()