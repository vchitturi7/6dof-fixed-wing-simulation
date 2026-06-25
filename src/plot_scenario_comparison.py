# plot_scenario_comparison.py
# Creates comparison plots from results/scenario_metrics.csv.
#
# This script is used after running:
#   python src/run_scenarios.py
#
# It reads the saved scenario metrics CSV and generates summary bar charts
# to compare controller performance across scenarios.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    """
    Loads the scenario metrics CSV file.

    Inputs:
        csv_path = path to scenario_metrics.csv

    Returns:
        metrics_df = pandas DataFrame containing scenario metrics
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find metrics CSV at: {csv_path}\n"
            "Run this first:\n"
            "    python src/run_scenarios.py"
        )

    metrics_df = pd.read_csv(csv_path)

    return metrics_df


def shorten_scenario_names(metrics_df):
    """
    Adds a shorter display name for plotting.
    """

    name_map = {
        "baseline_climb_no_gust": "Baseline\nNo Gust",
        "smooth_upward_gust": "Upward\nGust",
        "smooth_downward_gust": "Downward\nGust",
        "large_altitude_step": "Large\nAltitude Step",
        "higher_airspeed_climb": "Higher\nAirspeed",
    }

    metrics_df = metrics_df.copy()
    metrics_df["scenario_display_name"] = metrics_df["scenario_name"].map(name_map)

    # If a scenario is not in the map, use the original name.
    metrics_df["scenario_display_name"] = metrics_df["scenario_display_name"].fillna(
        metrics_df["scenario_name"]
    )

    return metrics_df


def save_figure(fig, output_path):
    """
    Saves a matplotlib figure.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def plot_metric_bar(metrics_df, metric_column, title, y_label, output_path):
    """
    Creates a bar chart for one metric across all scenarios.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        metrics_df["scenario_display_name"],
        metrics_df[metric_column],
    )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y")

    plt.tight_layout()

    save_figure(fig, output_path)

    return fig


def plot_combined_error_comparison(metrics_df, output_path):
    """
    Creates one combined figure comparing altitude and airspeed errors.
    """

    fig, ax = plt.subplots(figsize=(11, 6))

    x = range(len(metrics_df))
    width = 0.35

    x_altitude = [value - width / 2 for value in x]
    x_airspeed = [value + width / 2 for value in x]

    ax.bar(
        x_altitude,
        metrics_df["max_altitude_error_after_command_m"],
        width=width,
        label="Max Altitude Error [m]",
    )

    ax.bar(
        x_airspeed,
        metrics_df["max_airspeed_error_after_command_mps"],
        width=width,
        label="Max Airspeed Error [m/s]",
    )

    ax.set_title("Tracking Error Comparison Across Scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Error Magnitude")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df["scenario_display_name"])
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()

    save_figure(fig, output_path)

    return fig


def plot_combined_control_effort(metrics_df, output_path):
    """
    Creates one combined figure comparing elevator and throttle effort.

    Elevator is plotted in degrees.
    Throttle is scaled by 25 only for visual comparison,
    because throttle is unitless between 0 and 1.
    """

    fig, ax = plt.subplots(figsize=(11, 6))

    x = range(len(metrics_df))
    width = 0.35

    x_elevator = [value - width / 2 for value in x]
    x_throttle = [value + width / 2 for value in x]

    ax.bar(
        x_elevator,
        metrics_df["max_elevator_actual_deg"],
        width=width,
        label="Max Elevator Magnitude [deg]",
    )

    ax.bar(
        x_throttle,
        metrics_df["max_throttle_actual"] * 25.0,
        width=width,
        label="Max Throttle Command × 25",
    )

    ax.set_title("Control Effort Comparison Across Scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Control Effort")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df["scenario_display_name"])
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()

    save_figure(fig, output_path)

    return fig


def print_metric_summary(metrics_df):
    """
    Prints a quick text summary of the comparison results.
    """

    print("\nScenario comparison summary:")
    print("----------------------------")

    altitude_worst = metrics_df.loc[
        metrics_df["max_altitude_error_after_command_m"].idxmax()
    ]

    airspeed_worst = metrics_df.loc[
        metrics_df["max_airspeed_error_after_command_mps"].idxmax()
    ]

    elevator_worst = metrics_df.loc[
        metrics_df["max_elevator_actual_deg"].idxmax()
    ]

    throttle_worst = metrics_df.loc[
        metrics_df["max_throttle_command"].idxmax()
    ]

    print(
        "Largest altitude error: "
        f"{altitude_worst['scenario_name']} "
        f"({altitude_worst['max_altitude_error_after_command_m']:.3f} m)"
    )

    print(
        "Largest airspeed error: "
        f"{airspeed_worst['scenario_name']} "
        f"({airspeed_worst['max_airspeed_error_after_command_mps']:.3f} m/s)"
    )

    print(
        "Largest elevator command: "
        f"{elevator_worst['scenario_name']} "
        f"({elevator_worst['max_elevator_actual_deg']:.3f} deg)"
    )

    print(
        "Largest throttle command: "
        f"{throttle_worst['scenario_name']} "
        f"({throttle_worst['max_throttle_command']:.3f})"
    )


def main():
    """
    Main entry point.
    """

    metrics_csv_path = Path("results") / "scenario_metrics.csv"
    output_dir = Path("results") / "comparison_plots"

    metrics_df = load_metrics(metrics_csv_path)
    metrics_df = shorten_scenario_names(metrics_df)

    print_metric_summary(metrics_df)

    plot_metric_bar(
        metrics_df=metrics_df,
        metric_column="max_altitude_error_after_command_m",
        title="Maximum Altitude Error Across Scenarios",
        y_label="Max Altitude Error [m]",
        output_path=output_dir / "max_altitude_error.png",
    )

    plot_metric_bar(
        metrics_df=metrics_df,
        metric_column="max_airspeed_error_after_command_mps",
        title="Maximum Airspeed Error Across Scenarios",
        y_label="Max Airspeed Error [m/s]",
        output_path=output_dir / "max_airspeed_error.png",
    )

    plot_metric_bar(
        metrics_df=metrics_df,
        metric_column="max_elevator_actual_deg",
        title="Maximum Elevator Actual Across Scenarios",
        y_label="Max Elevator Actual [deg]",
        output_path=output_dir / "max_elevator_actual.png",
    )

    plot_metric_bar(
        metrics_df=metrics_df,
        metric_column="max_throttle_command",
        title="Maximum Throttle Command Across Scenarios",
        y_label="Max Throttle Command",
        output_path=output_dir / "max_throttle_command.png",
    )

    plot_combined_error_comparison(
        metrics_df=metrics_df,
        output_path=output_dir / "combined_tracking_error_comparison.png",
    )

    plot_combined_control_effort(
        metrics_df=metrics_df,
        output_path=output_dir / "combined_control_effort_comparison.png",
    )

    plt.show()


if __name__ == "__main__":
    main()