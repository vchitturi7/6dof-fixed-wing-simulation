# run_6dof_scenarios.py
# Runs multiple 6DOF open-loop simulation scenarios and saves metrics to CSV.
#
# Usage from project root:
#   python src/sixdof/run_6dof_scenarios.py
#
# This mirrors the longitudinal validation workflow:
#   scenario definitions -> simulation runs -> metrics -> CSV
#
# Current scenarios are open-loop:
#   1. trim hold
#   2. elevator pulse
#   3. aileron pulse
#   4. rudder pulse
#   5. throttle pulse
#
# Later, after closed-loop control is added, this will expand to:
#   - altitude hold
#   - airspeed hold
#   - heading hold
#   - waypoint following
#   - gust disturbance rejection

from pathlib import Path
import csv

from scenario_6dof_config import (
    Scenario6DOFConfig,
    DEFAULT_6DOF_SCENARIO,
    ELEVATOR_PULSE_SCENARIO,
    AILERON_PULSE_SCENARIO,
)
from simulate_6dof import run_simulation_6dof
from metrics_6dof import compute_6dof_metrics, print_6dof_metrics
from trim_6dof import solve_trim_6dof


def save_metrics_to_csv(metrics_list, output_path):
    """
    Saves a list of metrics dictionaries to a CSV file.
    """

    if not metrics_list:
        print("No 6DOF metrics to save.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics_list[0].keys())

    with output_path.open(mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            writer.writerow(metrics)

    print(f"\nSaved 6DOF metrics CSV: {output_path}")


def make_extra_6dof_scenarios():
    """
    Creates additional open-loop 6DOF scenarios.

    These are not closed-loop control tests yet.
    They are perturbation tests to verify that the 6DOF aircraft responds
    qualitatively correctly to control inputs.
    """

    rudder_pulse = Scenario6DOFConfig(
        scenario_name="sixdof_rudder_pulse",
        target_airspeed_mps=25.0,
        trim_altitude_m=100.0,
        sim_start_time_s=0.0,
        sim_end_time_s=30.0,
        num_time_points=2000,
        control_step_start_time_s=5.0,
        control_step_end_time_s=7.0,
        elevator_step_deg=0.0,
        aileron_step_deg=0.0,
        rudder_step_deg=5.0,
        throttle_step=0.0,
    )

    throttle_pulse = Scenario6DOFConfig(
        scenario_name="sixdof_throttle_pulse",
        target_airspeed_mps=25.0,
        trim_altitude_m=100.0,
        sim_start_time_s=0.0,
        sim_end_time_s=30.0,
        num_time_points=2000,
        control_step_start_time_s=5.0,
        control_step_end_time_s=10.0,
        elevator_step_deg=0.0,
        aileron_step_deg=0.0,
        rudder_step_deg=0.0,
        throttle_step=0.10,
    )

    return [
        rudder_pulse,
        throttle_pulse,
    ]


def main():
    """
    Runs all 6DOF open-loop validation scenarios.
    """

    scenarios = [
        DEFAULT_6DOF_SCENARIO,
        ELEVATOR_PULSE_SCENARIO,
        AILERON_PULSE_SCENARIO,
    ]

    scenarios.extend(make_extra_6dof_scenarios())

    all_metrics = []

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"Running 6DOF scenario: {scenario.scenario_name}")
        print("=" * 80)

        t, sol, data = run_simulation_6dof(
            config=scenario,
            show_plots=False,
        )

        # Re-solve trim here to get trim_info for metrics.
        # This is slightly redundant but keeps run_simulation_6dof simple.
        _, _, trim_info = solve_trim_6dof(
            target_airspeed_mps=scenario.target_airspeed_mps,
            altitude_m=scenario.trim_altitude_m,
        )

        metrics = compute_6dof_metrics(
            t=t,
            data=data,
            config=scenario,
            trim_info=trim_info,
        )

        print_6dof_metrics(metrics)
        all_metrics.append(metrics)

    save_metrics_to_csv(
        metrics_list=all_metrics,
        output_path="results/6dof/scenario_metrics_6dof.csv",
    )


if __name__ == "__main__":
    main()