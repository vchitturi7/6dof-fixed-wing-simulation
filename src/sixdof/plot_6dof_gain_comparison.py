# plot_6dof_gain_comparison.py
# Creates visual summaries of the 6DOF gain comparison campaign.
#
# Usage from project root:
#   python src/sixdof/plot_6dof_gain_comparison.py
#
# Input:
#   results/6dof/gain_comparison/gain_set_summary.csv
#
# Outputs:
#   results/6dof/gain_comparison/plots/mean_score_by_gain_set.png
#   results/6dof/gain_comparison/plots/mean_altitude_error_by_gain_set.png
#   results/6dof/gain_comparison/plots/mean_airspeed_error_by_gain_set.png
#   results/6dof/gain_comparison/plots/max_aileron_by_gain_set.png
#   results/6dof/gain_comparison/plots/max_rudder_by_gain_set.png
#   results/6dof/gain_comparison/plots/max_sideslip_by_gain_set.png
#
# Purpose:
#   These plots provide portfolio-ready visual evidence that controller tuning
#   improved the 6DOF autopilot by reducing lateral saturation while maintaining
#   strong tracking performance.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = Path("results/6dof/gain_comparison/gain_set_summary.csv")
OUTPUT_DIR = Path("results/6dof/gain_comparison/plots")


def load_gain_summary(input_csv=INPUT_CSV):
    """
    Loads the gain-set summary CSV.
    """

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find {input_csv}. "
            "Run python src/sixdof/run_6dof_gain_comparison.py first."
        )

    df = pd.read_csv(input_csv)

    required_columns = [
        "gain_set",
        "mean_score",
        "max_score",
        "mean_abs_final_altitude_error_m",
        "mean_abs_final_airspeed_error_mps",
        "mean_abs_final_heading_error_deg",
        "max_abs_aileron_deg_across_scenarios",
        "max_abs_rudder_deg_across_scenarios",
        "max_abs_beta_deg_across_scenarios",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Gain summary CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def make_gain_labels(df):
    """
    Makes cleaner labels for plotting.
    """

    label_map = {
        "original_baseline": "Original\nbaseline",
        "lateral_tuned": "Lateral\ntuned",
        "final_tuned": "Final\ntuned",
        "aggressive_longitudinal": "Aggressive\nlongitudinal",
    }

    return [
        label_map.get(gain_set, gain_set.replace("_", "\n"))
        for gain_set in df["gain_set"]
    ]


def save_bar_plot(
    df,
    y_column,
    title,
    y_label,
    output_filename,
    sort_by=None,
    lower_is_better=True,
):
    """
    Saves a single bar plot.

    Inputs:
        df              = gain summary dataframe
        y_column        = metric to plot
        title           = plot title
        y_label         = y-axis label
        output_filename = saved PNG filename
        sort_by         = optional column used for sorting
        lower_is_better = whether lower metric value is better
    """

    plot_df = df.copy()

    if sort_by is not None:
        plot_df = plot_df.sort_values(
            by=sort_by,
            ascending=lower_is_better,
        )

    labels = make_gain_labels(plot_df)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        labels,
        plot_df[y_column],
    )

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Gain Set")
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {output_path}")


def save_grouped_tracking_error_plot(df):
    """
    Saves a grouped bar plot comparing mean final tracking errors.

    Metrics:
        mean altitude error
        mean airspeed error
        mean heading error

    Note:
        These have different units, so this is mostly for visual comparison
        of relative controller behavior. The individual plots are more exact.
    """

    plot_df = df.copy()
    plot_df = plot_df.sort_values(
        by="mean_score",
        ascending=True,
    )

    labels = make_gain_labels(plot_df)

    x_positions = range(len(plot_df))
    bar_width = 0.25

    altitude_error = plot_df["mean_abs_final_altitude_error_m"]
    airspeed_error = plot_df["mean_abs_final_airspeed_error_mps"]
    heading_error = plot_df["mean_abs_final_heading_error_deg"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        [x - bar_width for x in x_positions],
        altitude_error,
        width=bar_width,
        label="Altitude error [m]",
    )

    ax.bar(
        list(x_positions),
        airspeed_error,
        width=bar_width,
        label="Airspeed error [m/s]",
    )

    ax.bar(
        [x + bar_width for x in x_positions],
        heading_error,
        width=bar_width,
        label="Heading error [deg]",
    )

    ax.set_title("Mean Final Tracking Error by Gain Set")
    ax.set_ylabel("Mean absolute final error")
    ax.set_xlabel("Gain Set")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "mean_tracking_errors_by_gain_set.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {output_path}")


def print_plot_summary(df):
    """
    Prints a concise interpretation of the gain comparison.
    """

    ranked_df = df.sort_values(
        by="mean_score",
        ascending=True,
    )

    best_score_row = ranked_df.iloc[0]

    best_altitude_row = df.sort_values(
        by="mean_abs_final_altitude_error_m",
        ascending=True,
    ).iloc[0]

    lowest_aileron_row = df.sort_values(
        by="max_abs_aileron_deg_across_scenarios",
        ascending=True,
    ).iloc[0]

    lowest_sideslip_row = df.sort_values(
        by="max_abs_beta_deg_across_scenarios",
        ascending=True,
    ).iloc[0]

    print("\nGain comparison interpretation:")
    print("--------------------------------")
    print(
        f"Best aggregate score: {best_score_row['gain_set']} "
        f"({best_score_row['mean_score']:.3f})"
    )
    print(
        f"Best mean altitude accuracy: {best_altitude_row['gain_set']} "
        f"({best_altitude_row['mean_abs_final_altitude_error_m']:.3f} m)"
    )
    print(
        f"Lowest max aileron usage: {lowest_aileron_row['gain_set']} "
        f"({lowest_aileron_row['max_abs_aileron_deg_across_scenarios']:.3f} deg)"
    )
    print(
        f"Lowest max sideslip: {lowest_sideslip_row['gain_set']} "
        f"({lowest_sideslip_row['max_abs_beta_deg_across_scenarios']:.3f} deg)"
    )


def main():
    """
    Main plotting workflow.
    """

    df = load_gain_summary()

    # Keep plots sorted by aggregate score unless the specific metric is the
    # plot's main ranking variable.
    score_sorted_df = df.sort_values(
        by="mean_score",
        ascending=True,
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="mean_score",
        title="Mean Aggregate Score by Gain Set",
        y_label="Mean score, lower is better",
        output_filename="mean_score_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="max_score",
        title="Worst-Case Scenario Score by Gain Set",
        y_label="Max score, lower is better",
        output_filename="max_score_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="mean_abs_final_altitude_error_m",
        title="Mean Final Altitude Error by Gain Set",
        y_label="Mean absolute final altitude error [m]",
        output_filename="mean_altitude_error_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="mean_abs_final_airspeed_error_mps",
        title="Mean Final Airspeed Error by Gain Set",
        y_label="Mean absolute final airspeed error [m/s]",
        output_filename="mean_airspeed_error_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="mean_abs_final_heading_error_deg",
        title="Mean Final Heading Error by Gain Set",
        y_label="Mean absolute final heading error [deg]",
        output_filename="mean_heading_error_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="max_abs_aileron_deg_across_scenarios",
        title="Maximum Aileron Usage Across Scenarios",
        y_label="Max absolute aileron deflection [deg]",
        output_filename="max_aileron_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="max_abs_rudder_deg_across_scenarios",
        title="Maximum Rudder Usage Across Scenarios",
        y_label="Max absolute rudder deflection [deg]",
        output_filename="max_rudder_by_gain_set.png",
    )

    save_bar_plot(
        df=score_sorted_df,
        y_column="max_abs_beta_deg_across_scenarios",
        title="Maximum Sideslip Across Scenarios",
        y_label="Max absolute sideslip [deg]",
        output_filename="max_sideslip_by_gain_set.png",
    )

    save_grouped_tracking_error_plot(
        df=score_sorted_df,
    )

    print_plot_summary(
        df=score_sorted_df,
    )


if __name__ == "__main__":
    main()