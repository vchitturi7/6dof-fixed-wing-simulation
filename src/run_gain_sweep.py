# run_gain_sweep.py
# Runs the same longitudinal scenario with multiple gain sets and ranks them
# by a weighted performance score.
#
# Saves results to: results/gain_sweep_metrics.csv
#
# Usage:
#   python src/run_gain_sweep.py

from pathlib import Path
import csv
from dataclasses import replace

from scenario_config import ScenarioConfig
from controller_config import ControllerGains
from simulate_longitudinal import run_simulation


# Reference scenario used for all gain sets.
# Same conditions as smooth_upward_gust scenario.
SWEEP_SCENARIO = ScenarioConfig(
    scenario_name="gain_sweep_reference",
    target_airspeed_mps=25.0,
    trim_altitude_m=100.0,
    altitude_step_m=20.0,
    command_start_time_s=5.0,
    sim_end_time_s=70.0,
    gust_start_time_s=25.0,
    gust_end_time_s=30.0,
    gust_wind_z_mps=2.0,
    use_smooth_gust=True,
    save_plots=False,
)

# Gain sets to compare.
# Each entry is (name, ControllerGains instance).
GAIN_SETS = [
    (
        "default",
        ControllerGains(),
    ),
    (
        "more_pitch_damping",
        ControllerGains(pitch_kd=0.60),
    ),
    (
        "lower_altitude_gain",
        # 0.125 deg/m instead of 0.25 deg/m — gentler pitch command per meter
        ControllerGains(altitude_kh_rad_per_m=0.0021816615649929116),
    ),
    (
        "gentler_integral",
        ControllerGains(pitch_ki=0.06),
    ),
    (
        "stronger_pitch_response",
        ControllerGains(pitch_kp=1.5, pitch_kd=0.50),
    ),
    (
        "stronger_airspeed_hold",
        ControllerGains(airspeed_kv=0.15),
    ),
]

# Weight applied to settling time in seconds (faster settling = lower score).
SETTLING_TIME_WEIGHT = 0.3

# Penalty added when the system never settles (equivalent to ~100 s of settling).
SETTLING_PENALTY = 30.0


def score_metrics(metrics):
    """
    Computes a scalar performance score. Lower is better.

    Weights reflect engineering priorities:
        altitude tracking > airspeed tracking > pitch tracking > control effort

    Settling time contributes continuously: every second of settling time adds
    SETTLING_TIME_WEIGHT to the score, so faster settling is always rewarded.
    If the system never settles, SETTLING_PENALTY is added instead.
    """

    altitude_penalty = 1.0 * metrics["max_altitude_error_after_command_m"]
    airspeed_penalty = 5.0 * metrics["max_airspeed_error_after_command_mps"]
    pitch_penalty    = 0.5 * metrics["max_pitch_error_after_command_deg"]
    elevator_penalty = 0.1 * metrics["max_elevator_actual_deg"]
    throttle_penalty = 5.0 * metrics["max_throttle_command"]

    alt_t = metrics["altitude_settling_time_s"]
    spd_t = metrics["airspeed_settling_time_s"]

    altitude_settling_score = (
        SETTLING_PENALTY if alt_t is None else SETTLING_TIME_WEIGHT * alt_t
    )
    airspeed_settling_score = (
        SETTLING_PENALTY if spd_t is None else SETTLING_TIME_WEIGHT * spd_t
    )

    return (
        altitude_penalty
        + airspeed_penalty
        + pitch_penalty
        + elevator_penalty
        + throttle_penalty
        + altitude_settling_score
        + airspeed_settling_score
    )


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

    print(f"\nSaved gain sweep metrics: {output_path}")


def print_sweep_summary(all_metrics):
    """
    Prints all gain sets ranked by score, best first.
    """

    sorted_metrics = sorted(all_metrics, key=lambda m: m["score"])

    print("\nGain sweep ranking (lower score = better):")
    print("-" * 55)

    for rank, m in enumerate(sorted_metrics, start=1):
        alt_err  = m["max_altitude_error_after_command_m"]
        spd_err  = m["max_airspeed_error_after_command_mps"]
        score    = m["score"]
        name     = m["gain_set_name"]
        print(
            f"  {rank}. {name:<28}  score={score:6.2f}"
            f"  alt_err={alt_err:.2f}m  spd_err={spd_err:.3f}m/s"
        )


def main():
    all_metrics = []

    for gain_set_name, gains in GAIN_SETS:
        print("\n" + "=" * 80)
        print(f"Running gain set: {gain_set_name}")
        print("=" * 80)

        scenario = replace(
            SWEEP_SCENARIO,
            scenario_name=f"gain_sweep_{gain_set_name}",
        )

        metrics = run_simulation(
            config=scenario,
            gains=gains,
            show_plots=False,
        )  

        metrics["gain_set_name"] = gain_set_name
        metrics["score"] = score_metrics(metrics)

        all_metrics.append(metrics)

    print_sweep_summary(all_metrics)

    save_metrics_to_csv(
        metrics_list=all_metrics,
        output_path="results/gain_sweep_metrics.csv",
    )


if __name__ == "__main__":
    main()
