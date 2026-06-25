# run_scenarios.py
# Runs multiple longitudinal GNC simulation scenarios and saves
# their performance metrics to a CSV file.
#
# This allows you to compare controller performance across:
#   - no gust
#   - upward gust
#   - downward gust
#   - larger altitude command
#   - higher airspeed command

from pathlib import Path
import csv
from controller_config import TUNED_GAINS
from scenario_config import ScenarioConfig
from simulate_longitudinal import run_simulation


def save_metrics_to_csv(metrics_list, output_path):
    """
    Saves a list of metrics dictionaries to a CSV file.
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

    print(f"\nSaved metrics CSV: {output_path}")


def main():
    scenarios = [
        ScenarioConfig(
            scenario_name="baseline_climb_no_gust",
            target_airspeed_mps=25.0,
            trim_altitude_m=100.0,
            altitude_step_m=20.0,
            command_start_time_s=5.0,
            sim_end_time_s=70.0,
            gust_wind_z_mps=0.0,
            use_smooth_gust=True,
        ),

        ScenarioConfig(
            scenario_name="smooth_upward_gust",
            target_airspeed_mps=25.0,
            trim_altitude_m=100.0,
            altitude_step_m=20.0,
            command_start_time_s=5.0,
            sim_end_time_s=70.0,
            gust_start_time_s=25.0,
            gust_end_time_s=30.0,
            gust_wind_z_mps=2.0,
            use_smooth_gust=True,
        ),

        ScenarioConfig(
            scenario_name="smooth_downward_gust",
            target_airspeed_mps=25.0,
            trim_altitude_m=100.0,
            altitude_step_m=20.0,
            command_start_time_s=5.0,
            sim_end_time_s=70.0,
            gust_start_time_s=25.0,
            gust_end_time_s=30.0,
            gust_wind_z_mps=-2.0,
            use_smooth_gust=True,
        ),

        ScenarioConfig(
            scenario_name="large_altitude_step",
            target_airspeed_mps=25.0,
            trim_altitude_m=100.0,
            altitude_step_m=40.0,
            command_start_time_s=5.0,
            sim_end_time_s=90.0,
            gust_start_time_s=35.0,
            gust_end_time_s=40.0,
            gust_wind_z_mps=2.0,
            use_smooth_gust=True,
        ),

        ScenarioConfig(
            scenario_name="higher_airspeed_climb",
            target_airspeed_mps=30.0,
            trim_altitude_m=100.0,
            altitude_step_m=20.0,
            command_start_time_s=5.0,
            sim_end_time_s=70.0,
            gust_start_time_s=25.0,
            gust_end_time_s=30.0,
            gust_wind_z_mps=2.0,
            use_smooth_gust=True,
        ),
    ]

    all_metrics = []

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"Running scenario: {scenario.scenario_name}")
        print("=" * 80)

        metrics = run_simulation(config = scenario, gains = TUNED_GAINS, show_plots=False,)
        all_metrics.append(metrics)

    save_metrics_to_csv(
        metrics_list=all_metrics,
        output_path="results/scenario_metrics.csv",
    )


if __name__ == "__main__":
    main()