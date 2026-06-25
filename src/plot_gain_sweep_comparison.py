# plot_gain_sweep_comparison.py
# Reads results/gain_sweep_metrics.csv and produces bar charts comparing
# controller performance across all gain sets.
#
# Run after:
#   python src/run_gain_sweep.py
#
# Usage:
#   python src/plot_gain_sweep_comparison.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find gain sweep CSV at: {csv_path}\n"
            "Run this first:\n"
            "    python src/run_gain_sweep.py"
        )

    return pd.read_csv(csv_path)


def save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def plot_bar(df, x_col, y_col, title, y_label, output_path, color=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(df[x_col], df[y_col], color=color)
    ax.set_title(title)
    ax.set_xlabel("Gain Set")
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig


def plot_settling_times(df, output_path):
    """
    Side-by-side bars for altitude and airspeed settling times.
    Missing values (never settled) are shown as a fixed high bar with a label.
    """

    NEVER_SETTLED_DISPLAY = 70.0

    alt_times = df["altitude_settling_time_s"].fillna(NEVER_SETTLED_DISPLAY)
    spd_times = df["airspeed_settling_time_s"].fillna(NEVER_SETTLED_DISPLAY)

    x = range(len(df))
    width = 0.35
    x_alt = [v - width / 2 for v in x]
    x_spd = [v + width / 2 for v in x]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_alt = ax.bar(x_alt, alt_times, width=width, label="Altitude Settling [s]")
    bars_spd = ax.bar(x_spd, spd_times, width=width, label="Airspeed Settling [s]")

    # Label bars that represent "never settled"
    for bar, raw in zip(bars_alt, df["altitude_settling_time_s"]):
        if pd.isna(raw):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                "N/S",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar, raw in zip(bars_spd, df["airspeed_settling_time_s"]):
        if pd.isna(raw):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                "N/S",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Settling Times Across Gain Sets  (N/S = never settled)")
    ax.set_xlabel("Gain Set")
    ax.set_ylabel("Settling Time [s]")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["gain_set_name"], rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig


def plot_score_ranked(df, output_path):
    """
    Bar chart of total score, sorted best (lowest) to worst.
    """

    df_sorted = df.sort_values("score")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(df_sorted["gain_set_name"], df_sorted["score"])

    # Annotate each bar with its score value
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Overall Performance Score by Gain Set  (lower = better)")
    ax.set_xlabel("Gain Set")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig


def print_summary(df):
    print("\nGain sweep comparison summary:")
    print("-------------------------------")

    best = df.loc[df["score"].idxmin()]
    worst = df.loc[df["score"].idxmax()]

    print(f"Best overall score:   {best['gain_set_name']}  (score = {best['score']:.2f})")
    print(f"Worst overall score:  {worst['gain_set_name']}  (score = {worst['score']:.2f})")

    best_alt = df.loc[df["max_altitude_error_after_command_m"].idxmin()]
    best_spd = df.loc[df["max_airspeed_error_after_command_mps"].idxmin()]

    print(
        f"Best altitude tracking:  {best_alt['gain_set_name']}"
        f"  ({best_alt['max_altitude_error_after_command_m']:.2f} m)"
    )
    print(
        f"Best airspeed tracking:  {best_spd['gain_set_name']}"
        f"  ({best_spd['max_airspeed_error_after_command_mps']:.3f} m/s)"
    )


def main():
    csv_path = Path("results") / "gain_sweep_metrics.csv"
    output_dir = Path("results") / "gain_sweep_plots"

    df = load_metrics(csv_path)

    print_summary(df)

    plot_score_ranked(
        df=df,
        output_path=output_dir / "score_ranked.png",
    )

    plot_bar(
        df=df,
        x_col="gain_set_name",
        y_col="max_altitude_error_after_command_m",
        title="Max Altitude Error by Gain Set",
        y_label="Max Altitude Error [m]",
        output_path=output_dir / "max_altitude_error.png",
    )

    plot_bar(
        df=df,
        x_col="gain_set_name",
        y_col="max_airspeed_error_after_command_mps",
        title="Max Airspeed Error by Gain Set",
        y_label="Max Airspeed Error [m/s]",
        output_path=output_dir / "max_airspeed_error.png",
    )

    plot_bar(
        df=df,
        x_col="gain_set_name",
        y_col="max_elevator_actual_deg",
        title="Max Actual Elevator Deflection by Gain Set",
        y_label="Max Elevator  [deg]",
        output_path=output_dir / "max_elevator_actual.png",
    )

    plot_bar(
        df=df,
        x_col="gain_set_name",
        y_col="max_throttle_command",
        title="Max Throttle Command by Gain Set",
        y_label="Max Throttle (0–1)",
        output_path=output_dir / "max_throttle.png",
    )

    plot_settling_times(
        df=df,
        output_path=output_dir / "settling_times.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
