# run_6dof_closed_loop_scenarios.py
# Runs multiple closed-loop 6DOF autopilot scenarios and saves metrics to CSV.
#
# Usage from project root:
#   python src/sixdof/run_6dof_closed_loop_scenarios.py
#
# This is the closed-loop 6DOF validation workflow:
#   scenario definitions -> closed-loop simulation -> metrics -> CSV
#
# Current scenario types:
#   1. altitude step only
#   2. heading step only
#   3. combined altitude + heading step
#   4. airspeed step
#   5. larger combined maneuver

from pathlib import Path
import csv

from simulate_6dof_closed_loop import (
    ClosedLoop6DOFScenario,
    run_closed_loop_6dof,
)
from controllers_6dof import DEFAULT_6DOF_GAINS
from metrics_6dof_closed_loop import (
    compute_closed_loop_6dof_metrics,
    print_closed_loop_6dof_metrics,
)


def save_metrics_to_csv(metrics_list, output_path):
    """
    Saves a list of metrics dictionaries to a CSV file.
    """

    if not metrics_list:
        print("No closed-loop 6DOF metrics to save.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics_list[0].keys())

    with output_path.open(mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            writer.writerow(metrics)

    print(f"\nSaved closed-loop 6DOF metrics CSV: {output_path}")


def make_scenario(
    scenario_name,
    altitude_step_m,
    heading_step_deg,
    airspeed_step_mps,
    sim_end_time_s=60.0,
):
    """
    Creates a closed-loop 6DOF scenario from simple maneuver inputs.
    """

    scenario = ClosedLoop6DOFScenario()

    scenario.scenario_name = scenario_name
    scenario.altitude_step_m = altitude_step_m
    scenario.heading_step_deg = heading_step_deg
    scenario.airspeed_step_mps = airspeed_step_mps
    scenario.sim_end_time_s = sim_end_time_s
    scenario.save_plots = True
    scenario.results_dir = "results/6dof/closed_loop"

    return scenario


def main():
    """
    Runs all closed-loop 6DOF validation scenarios.
    """

    scenarios = [
        make_scenario(
            scenario_name="closed_loop_altitude_step",
            altitude_step_m=20.0,
            heading_step_deg=0.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="closed_loop_heading_step",
            altitude_step_m=0.0,
            heading_step_deg=20.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="closed_loop_altitude_heading_step",
            altitude_step_m=20.0,
            heading_step_deg=20.0,
            airspeed_step_mps=0.0,
            sim_end_time_s=70.0,
        ),
        make_scenario(
            scenario_name="closed_loop_airspeed_step",
            altitude_step_m=0.0,
            heading_step_deg=0.0,
            airspeed_step_mps=3.0,
            sim_end_time_s=60.0,
        ),
        make_scenario(
            scenario_name="closed_loop_large_combined_step",
            altitude_step_m=40.0,
            heading_step_deg=35.0,
            airspeed_step_mps=3.0,
            sim_end_time_s=90.0,
        ),
    ]

    all_metrics = []

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"Running closed-loop 6DOF scenario: {scenario.scenario_name}")
        print("=" * 80)

        t, sol, data = run_closed_loop_6dof(
            config=scenario,
            gains=DEFAULT_6DOF_GAINS,
            show_plots=False,
        )

        metrics = compute_closed_loop_6dof_metrics(
            t=t,
            data=data,
            config=scenario,
        )

        print_closed_loop_6dof_metrics(metrics)

        all_metrics.append(metrics)

    save_metrics_to_csv(
        metrics_list=all_metrics,
        output_path="results/6dof/closed_loop_metrics.csv",
    )


if __name__ == "__main__":
    main()