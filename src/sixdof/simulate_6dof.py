# simulate_6dof.py
# Open-loop 6DOF simulation for the fixed-wing aircraft.
#
# This script:
#   1. Solves for straight-and-level 6DOF trim.
#   2. Integrates the 12-state 6DOF dynamics.
#   3. Applies optional small open-loop control pulses.
#   4. Plots trajectory, attitude, airspeed, rates, and controls.
#
# This is not closed-loop autopilot yet.
# It is the first full 6DOF simulation milestone.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from scenario_6dof_config import (
    DEFAULT_6DOF_SCENARIO,
    ELEVATOR_PULSE_SCENARIO,
    AILERON_PULSE_SCENARIO,
    Scenario6DOFConfig,
)
from trim_6dof import solve_trim_6dof
from dynamics_6dof import dynamics_6dof, wind_body_from_controls
from aerodynamics_6dof import compute_air_data


def save_figure(fig, config, filename):
    """
    Saves a matplotlib figure if config.save_plots is True.
    """

    if not config.save_plots:
        return

    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure: {output_path}")


def build_controls_function(trim_controls, config):
    """
    Builds an open-loop controls function.

    The baseline controls are trim controls.
    During the control step window, optional perturbations are added.
    """

    delta_e_trim = trim_controls["delta_e"]
    delta_a_trim = trim_controls["delta_a"]
    delta_r_trim = trim_controls["delta_r"]
    delta_t_trim = trim_controls["delta_t"]

    elevator_step_rad = np.deg2rad(config.elevator_step_deg)
    aileron_step_rad = np.deg2rad(config.aileron_step_deg)
    rudder_step_rad = np.deg2rad(config.rudder_step_deg)
    throttle_step = config.throttle_step

    wind_inertial = np.array([
        config.wind_x_inertial_mps,
        config.wind_y_inertial_mps,
        config.wind_z_down_inertial_mps,
    ])

    def controls(t, state):
        in_step_window = (
            config.control_step_start_time_s
            <= t
            <= config.control_step_end_time_s
        )

        if in_step_window:
            delta_e = delta_e_trim + elevator_step_rad
            delta_a = delta_a_trim + aileron_step_rad
            delta_r = delta_r_trim + rudder_step_rad
            delta_t = delta_t_trim + throttle_step
        else:
            delta_e = delta_e_trim
            delta_a = delta_a_trim
            delta_r = delta_r_trim
            delta_t = delta_t_trim

        delta_t = max(min(delta_t, 1.0), 0.0)

        return {
            "delta_e": delta_e,
            "delta_a": delta_a,
            "delta_r": delta_r,
            "delta_t": delta_t,
            "wind_inertial": wind_inertial,
        }

    return controls


def post_process_solution(t, sol_y, controls_function):
    """
    Computes useful derived quantities from the 6DOF simulation.
    """

    x = sol_y[0]
    y = sol_y[1]
    z_down = sol_y[2]

    u = sol_y[3]
    v = sol_y[4]
    w = sol_y[5]

    phi = sol_y[6]
    theta = sol_y[7]
    psi = sol_y[8]

    p_rate = sol_y[9]
    q_rate = sol_y[10]
    r_rate = sol_y[11]

    altitude = -z_down

    airspeed = []
    alpha = []
    beta = []

    delta_e = []
    delta_a = []
    delta_r = []
    delta_t = []

    for i, time in enumerate(t):
        state_i = sol_y[:, i]
        controls_i = controls_function(time, state_i)

        phi_i = state_i[6]
        theta_i = state_i[7]
        psi_i = state_i[8]

        velocity_body_i = np.array([
            state_i[3],
            state_i[4],
            state_i[5],
        ])

        wind_body_i = wind_body_from_controls(
            phi=phi_i,
            theta=theta_i,
            psi=psi_i,
            controls=controls_i,
        )

        air_data_i = compute_air_data(
            velocity_body=velocity_body_i,
            wind_body=wind_body_i,
        )

        airspeed.append(air_data_i["V"])
        alpha.append(air_data_i["alpha"])
        beta.append(air_data_i["beta"])

        delta_e.append(controls_i["delta_e"])
        delta_a.append(controls_i["delta_a"])
        delta_r.append(controls_i["delta_r"])
        delta_t.append(controls_i["delta_t"])

    return {
        "x": x,
        "y": y,
        "z_down": z_down,
        "altitude": altitude,
        "u": u,
        "v": v,
        "w": w,
        "phi": phi,
        "theta": theta,
        "psi": psi,
        "p_rate": p_rate,
        "q_rate": q_rate,
        "r_rate": r_rate,
        "airspeed": np.array(airspeed),
        "alpha": np.array(alpha),
        "beta": np.array(beta),
        "delta_e": np.array(delta_e),
        "delta_a": np.array(delta_a),
        "delta_r": np.array(delta_r),
        "delta_t": np.array(delta_t),
    }


def print_trim_summary(trim_info):
    """
    Prints trim solution summary.
    """

    print("\n6DOF trim solution:")
    print("-------------------")
    print(f"Target airspeed: {trim_info['target_airspeed_mps']:.3f} m/s")
    print(f"Altitude:        {trim_info['altitude_m']:.3f} m")
    print(f"Alpha trim:      {trim_info['alpha_trim_deg']:.3f} deg")
    print(f"Theta trim:      {trim_info['theta_trim_deg']:.3f} deg")
    print(f"Elevator trim:   {trim_info['delta_e_trim_deg']:.3f} deg")
    print(f"Throttle trim:   {trim_info['delta_t_trim']:.3f}")


def print_simulation_summary(data):
    """
    Prints quick final-state summary.
    """

    print("\n6DOF simulation summary:")
    print("------------------------")
    print(f"Final x position:     {data['x'][-1]:.3f} m")
    print(f"Final y position:     {data['y'][-1]:.3f} m")
    print(f"Final altitude:       {data['altitude'][-1]:.3f} m")
    print(f"Final airspeed:       {data['airspeed'][-1]:.3f} m/s")
    print(f"Final roll angle:     {np.rad2deg(data['phi'][-1]):.3f} deg")
    print(f"Final pitch angle:    {np.rad2deg(data['theta'][-1]):.3f} deg")
    print(f"Final heading angle:  {np.rad2deg(data['psi'][-1]):.3f} deg")
    print(f"Max abs roll angle:   {np.max(np.abs(np.rad2deg(data['phi']))):.3f} deg")
    print(f"Max abs pitch angle:  {np.max(np.abs(np.rad2deg(data['theta']))):.3f} deg")
    print(f"Max abs beta:         {np.max(np.abs(np.rad2deg(data['beta']))):.3f} deg")


def plot_trajectory(data, config):
    """
    Plots 3D trajectory.
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        data["x"],
        data["y"],
        data["altitude"],
        label="Aircraft trajectory",
    )

    ax.set_title("6DOF Open-Loop Trajectory")
    ax.set_xlabel("x / North [m]")
    ax.set_ylabel("y / East [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend()

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_trajectory_3d.png",
    )


def plot_response(t, data, config):
    """
    Plots 6DOF response time histories.
    """

    fig, axs = plt.subplots(4, 2, figsize=(14, 13))
    fig.suptitle("6DOF Open-Loop Aircraft Response", fontsize=14)

    axs[0, 0].plot(t, data["altitude"])
    axs[0, 0].set_title("Altitude")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Altitude [m]")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, data["airspeed"])
    axs[0, 1].set_title("Airspeed")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("V [m/s]")
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(data["phi"]), label="Roll φ")
    axs[1, 0].plot(t, np.rad2deg(data["theta"]), label="Pitch θ")
    axs[1, 0].plot(t, np.rad2deg(data["psi"]), label="Heading ψ")
    axs[1, 0].set_title("Euler Angles")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Angle [deg]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(data["alpha"]), label="AoA α")
    axs[1, 1].plot(t, np.rad2deg(data["beta"]), label="Sideslip β")
    axs[1, 1].set_title("Air Data Angles")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Angle [deg]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, np.rad2deg(data["p_rate"]), label="p")
    axs[2, 0].plot(t, np.rad2deg(data["q_rate"]), label="q")
    axs[2, 0].plot(t, np.rad2deg(data["r_rate"]), label="r")
    axs[2, 0].set_title("Body Rates")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Rate [deg/s]")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].plot(t, data["u"], label="u")
    axs[2, 1].plot(t, data["v"], label="v")
    axs[2, 1].plot(t, data["w"], label="w")
    axs[2, 1].set_title("Body Velocities")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Velocity [m/s]")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    axs[3, 0].plot(t, np.rad2deg(data["delta_e"]), label="Elevator")
    axs[3, 0].plot(t, np.rad2deg(data["delta_a"]), label="Aileron")
    axs[3, 0].plot(t, np.rad2deg(data["delta_r"]), label="Rudder")
    axs[3, 0].set_title("Control Surface Inputs")
    axs[3, 0].set_xlabel("Time [s]")
    axs[3, 0].set_ylabel("Deflection [deg]")
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    axs[3, 1].plot(t, data["delta_t"], label="Throttle")
    axs[3, 1].set_title("Throttle Input")
    axs[3, 1].set_xlabel("Time [s]")
    axs[3, 1].set_ylabel("Throttle [0–1]")
    axs[3, 1].legend()
    axs[3, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_response.png",
    )


def run_simulation_6dof(
    config: Scenario6DOFConfig,
    show_plots=True,
):
    """
    Runs the 6DOF open-loop simulation.

    Returns:
        t
        sol
        data
    """

    print("\n6DOF scenario:")
    print("--------------")
    print(f"Scenario name:      {config.scenario_name}")
    print(f"Target airspeed:    {config.target_airspeed_mps:.3f} m/s")
    print(f"Trim altitude:      {config.trim_altitude_m:.3f} m")
    print(f"Simulation time:    {config.sim_start_time_s:.3f} to {config.sim_end_time_s:.3f} s")
    print(f"Elevator step:      {config.elevator_step_deg:.3f} deg")
    print(f"Aileron step:       {config.aileron_step_deg:.3f} deg")
    print(f"Rudder step:        {config.rudder_step_deg:.3f} deg")
    print(f"Throttle step:      {config.throttle_step:.3f}")

    trim_state, trim_controls, trim_info = solve_trim_6dof(
        target_airspeed_mps=config.target_airspeed_mps,
        altitude_m=config.trim_altitude_m,
    )

    print_trim_summary(trim_info)

    controls_function = build_controls_function(
        trim_controls=trim_controls,
        config=config,
    )

    def dynamics_wrapper(t, state):
        controls = controls_function(t, state)

        return dynamics_6dof(
            t=t,
            state=state,
            controls=controls,
        )

    t_span = (
        config.sim_start_time_s,
        config.sim_end_time_s,
    )

    t_eval = np.linspace(
        config.sim_start_time_s,
        config.sim_end_time_s,
        config.num_time_points,
    )

    sol = solve_ivp(
        dynamics_wrapper,
        t_span,
        trim_state,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"6DOF simulation failed: {sol.message}")

    t = sol.t
    data = post_process_solution(
        t=t,
        sol_y=sol.y,
        controls_function=controls_function,
    )

    print_simulation_summary(data)

    plot_trajectory(
        data=data,
        config=config,
    )

    plot_response(
        t=t,
        data=data,
        config=config,
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return t, sol, data


def main():
    """
    Entry point for manual 6DOF simulation.

    Change scenario here to test different open-loop cases:

        DEFAULT_6DOF_SCENARIO
        ELEVATOR_PULSE_SCENARIO
        AILERON_PULSE_SCENARIO
    """

    config = AILERON_PULSE_SCENARIO

    run_simulation_6dof(
        config=config,
        show_plots=True,
    )


if __name__ == "__main__":
    main()