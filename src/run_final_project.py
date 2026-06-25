# run_final_project.py
# Final validation runner for the 6DOF fixed-wing GNC project.
#
# Usage from project root:
#   python run_final_project.py
#
# This script runs the final project validation sequence:
#
#   1. Rotation utility tests
#   2. Aerodynamic model tests
#   3. 6DOF dynamics tests
#   4. Controller tests
#   5. Open-loop 6DOF scenario batch
#   6. Closed-loop 6DOF scenario batch
#   7. Gain comparison campaign
#   8. Gain comparison plots
#   9. Final waypoint-following mission
#   10. Waypoint-following metrics CSV
#
# It is meant to be the final "one button" project runner.

import sys
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent
SIXDOF_DIR = PROJECT_ROOT / "sixdof"


def run_command(command, description):
    """
    Runs a command and stops if it fails.
    """

    print("\n" + "=" * 90)
    print(description)
    print("=" * 90)
    print("Command:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed during: {description}"
        )


def run_python_script(script_path, description):
    """
    Runs a Python script from the project root.
    """

    run_command(
        command=[
            sys.executable,
            str(script_path),
        ],
        description=description,
    )


def run_waypoint_mission_with_metrics():
    """
    Runs final waypoint-following mission and saves metrics.

    This is done inside Python instead of subprocess so we can access the
    returned t/state/data objects and compute metrics directly.
    """

    print("\n" + "=" * 90)
    print("Running final waypoint-following mission with metrics")
    print("=" * 90)

    sys.path.insert(
        0,
        str(SIXDOF_DIR),
    )

    from simulate_6dof_waypoint_following import (
        WaypointFollowingScenario,
        FINAL_TUNED_6DOF_GAINS,
        run_waypoint_following_6dof,
    )
    from waypoint_guidance import make_default_waypoints
    from metrics_6dof_waypoint import (
        compute_waypoint_metrics,
        print_waypoint_metrics,
        save_waypoint_metrics_to_csv,
    )

    scenario = WaypointFollowingScenario()

    t, state_history, data = run_waypoint_following_6dof(
        scenario=scenario,
        gains=FINAL_TUNED_6DOF_GAINS,
        show_plots=False,
    )

    waypoints = make_default_waypoints()

    metrics = compute_waypoint_metrics(
        t=t,
        data=data,
        waypoints=waypoints,
        scenario=scenario,
    )

    print_waypoint_metrics(metrics)

    save_waypoint_metrics_to_csv(
        metrics=metrics,
        output_path=PROJECT_ROOT / "results" / "6dof" / "waypoint_following" / "waypoint_metrics.csv",
    )


def main():
    """
    Runs final project validation workflow.
    """

    print("\nStarting final 6DOF fixed-wing GNC project validation...")
    print(f"Project root: {PROJECT_ROOT}")

    run_python_script(
        script_path=SIXDOF_DIR / "test_rotations.py",
        description="1. Running rotation utility tests",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "test_aerodynamics_6dof.py",
        description="2. Running aerodynamic model tests",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "test_dynamics_6dof.py",
        description="3. Running 6DOF dynamics tests",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "test_controllers_6dof.py",
        description="4. Running 6DOF controller tests",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "run_6dof_scenarios.py",
        description="5. Running open-loop 6DOF scenarios",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "run_6dof_closed_loop_scenarios.py",
        description="6. Running closed-loop 6DOF scenarios",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "run_6dof_gain_comparison.py",
        description="7. Running 6DOF gain comparison campaign",
    )

    run_python_script(
        script_path=SIXDOF_DIR / "plot_6dof_gain_comparison.py",
        description="8. Plotting 6DOF gain comparison results",
    )

    run_waypoint_mission_with_metrics()

    print("\n" + "=" * 90)
    print("Final project validation complete.")
    print("=" * 90)

    print("\nKey outputs:")
    print("  results/6dof/scenario_metrics_6dof.csv")
    print("  results/6dof/closed_loop_metrics.csv")
    print("  results/6dof/gain_comparison/gain_set_summary.csv")
    print("  results/6dof/gain_comparison/plots/")
    print("  results/6dof/waypoint_following/")
    print("  results/6dof/waypoint_following/waypoint_metrics.csv")


if __name__ == "__main__":
    main()