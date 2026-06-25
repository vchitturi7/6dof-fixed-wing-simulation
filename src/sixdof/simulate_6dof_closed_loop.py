# simulate_6dof_closed_loop.py
# First closed-loop 6DOF simulation for the fixed-wing aircraft.
#
# This script:
#   1. Solves for straight-and-level 6DOF trim.
#   2. Defines altitude, airspeed, and heading commands.
#   3. Uses the 6DOF autopilot to compute elevator, aileron, rudder, and throttle.
#   4. Integrates the 12-state 6DOF dynamics.
#   5. Plots tracking response, attitude, rates, controls, and 3D trajectory.
#
# This is the first true 6DOF GNC simulation milestone.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from trim_6dof import solve_trim_6dof
from dynamics_6dof import dynamics_6dof, wind_body_from_controls
from aerodynamics_6dof import compute_air_data
from controllers_6dof import (
    DEFAULT_6DOF_GAINS,
    Controller6DOFGains,
    full_6dof_autopilot,
)


class ClosedLoop6DOFScenario:
    """
    Simple closed-loop 6DOF scenario configuration.

    This is intentionally a normal class instead of a dataclass for now,
    so the file stays standalone and easy to modify.
    """

    def __init__(self):
        # Scenario name
        self.scenario_name = "sixdof_closed_loop_altitude_heading_hold"

        # Trim / initial condition
        self.target_airspeed_mps = 25.0
        self.trim_altitude_m = 100.0

        # Simulation time
        self.sim_start_time_s = 0.0
        self.sim_end_time_s = 60.0
        self.num_time_points = 3000

        # Command timing
        self.command_start_time_s = 5.0

        # Commands after command_start_time_s
        self.altitude_step_m = 20.0
        self.heading_step_deg = 20.0
        self.airspeed_step_mps = 0.0

        # Constant inertial wind in NED coordinates
        self.wind_x_inertial_mps = 0.0
        self.wind_y_inertial_mps = 0.0
        self.wind_z_down_inertial_mps = 0.0

        # Output
        self.save_plots = True
        self.results_dir = "results/6dof/closed_loop"


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


def get_command_values(t, config, trim_info):
    """
    Defines altitude, airspeed, and heading commands.

    Before command_start_time_s:
        hold trim altitude, trim airspeed, and zero heading.

    After command_start_time_s:
        command altitude step, optional airspeed step, and heading step.
    """

    altitude_trim = config.trim_altitude_m
    airspeed_trim = config.target_airspeed_mps
    heading_trim_rad = 0.0

    if t < config.command_start_time_s:
        altitude_command_m = altitude_trim
        airspeed_command_mps = airspeed_trim
        heading_command_rad = heading_trim_rad
    else:
        altitude_command_m = altitude_trim + config.altitude_step_m
        airspeed_command_mps = airspeed_trim + config.airspeed_step_mps
        heading_command_rad = heading_trim_rad + np.deg2rad(config.heading_step_deg)

    return {
        "altitude_command_m": altitude_command_m,
        "airspeed_command_mps": airspeed_command_mps,
        "heading_command_rad": heading_command_rad,
    }


def get_wind_inertial(config):
    """
    Returns constant inertial-frame wind vector in NED coordinates.

    Convention:
        +x = north/forward
        +y = east/right
        +z = down
    """

    return np.array([
        config.wind_x_inertial_mps,
        config.wind_y_inertial_mps,
        config.wind_z_down_inertial_mps,
    ])


def compute_air_data_from_state(state, controls):
    """
    Computes air data for a 6DOF state and control dictionary.

    This converts inertial wind into body wind, then computes:
        V, alpha, beta
    """

    phi = state[6]
    theta = state[7]
    psi = state[8]

    velocity_body = np.array([
        state[3],
        state[4],
        state[5],
    ])

    wind_body = wind_body_from_controls(
        phi=phi,
        theta=theta,
        psi=psi,
        controls=controls,
    )

    air_data = compute_air_data(
        velocity_body=velocity_body,
        wind_body=wind_body,
    )

    return air_data


def build_closed_loop_controls_function(trim_controls, trim_info, config, gains):
    """
    Builds a closed-loop controls function.

    At each timestep:
        1. Compute current air data.
        2. Build altitude/airspeed/heading command.
        3. Call full_6dof_autopilot.
        4. Return controls for dynamics.
    """

    wind_inertial = get_wind_inertial(config)

    def controls(t, state):
        # Baseline controls include wind so air data uses the same wind
        # as the aircraft dynamics.
        controls_for_air_data = {
            "delta_e": trim_controls["delta_e"],
            "delta_a": trim_controls["delta_a"],
            "delta_r": trim_controls["delta_r"],
            "delta_t": trim_controls["delta_t"],
            "wind_inertial": wind_inertial,
        }

        air_data = compute_air_data_from_state(
            state=state,
            controls=controls_for_air_data,
        )

        commands = get_command_values(
            t=t,
            config=config,
            trim_info=trim_info,
        )

        autopilot_controls = full_6dof_autopilot(
            commands=commands,
            state=state,
            air_data=air_data,
            trim_controls=trim_controls,
            trim_info=trim_info,
            gains=gains,
        )

        return {
            "delta_e": autopilot_controls["delta_e"],
            "delta_a": autopilot_controls["delta_a"],
            "delta_r": autopilot_controls["delta_r"],
            "delta_t": autopilot_controls["delta_t"],
            "theta_command_rad": autopilot_controls["theta_command_rad"],
            "phi_command_rad": autopilot_controls["phi_command_rad"],
            "altitude_command_m": commands["altitude_command_m"],
            "airspeed_command_mps": commands["airspeed_command_mps"],
            "heading_command_rad": commands["heading_command_rad"],
            "wind_inertial": wind_inertial,
        }

    return controls


def post_process_solution(t, sol_y, controls_function):
    """
    Computes useful derived quantities from closed-loop 6DOF simulation.
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

    theta_command = []
    phi_command = []
    altitude_command = []
    airspeed_command = []
    heading_command = []

    for i, time in enumerate(t):
        state_i = sol_y[:, i]
        controls_i = controls_function(time, state_i)

        air_data_i = compute_air_data_from_state(
            state=state_i,
            controls=controls_i,
        )

        airspeed.append(air_data_i["V"])
        alpha.append(air_data_i["alpha"])
        beta.append(air_data_i["beta"])

        delta_e.append(controls_i["delta_e"])
        delta_a.append(controls_i["delta_a"])
        delta_r.append(controls_i["delta_r"])
        delta_t.append(controls_i["delta_t"])

        theta_command.append(controls_i["theta_command_rad"])
        phi_command.append(controls_i["phi_command_rad"])
        altitude_command.append(controls_i["altitude_command_m"])
        airspeed_command.append(controls_i["airspeed_command_mps"])
        heading_command.append(controls_i["heading_command_rad"])

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
        "theta_command": np.array(theta_command),
        "phi_command": np.array(phi_command),
        "altitude_command": np.array(altitude_command),
        "airspeed_command": np.array(airspeed_command),
        "heading_command": np.array(heading_command),
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


def print_simulation_summary(data, config):
    """
    Prints quick final-state summary.
    """

    altitude_error_final = data["altitude_command"][-1] - data["altitude"][-1]
    airspeed_error_final = data["airspeed_command"][-1] - data["airspeed"][-1]
    heading_error_final_deg = np.rad2deg(
        data["heading_command"][-1] - data["psi"][-1]
    )

    print("\nClosed-loop 6DOF simulation summary:")
    print("------------------------------------")
    print(f"Final x position:        {data['x'][-1]:.3f} m")
    print(f"Final y position:        {data['y'][-1]:.3f} m")
    print(f"Final altitude:          {data['altitude'][-1]:.3f} m")
    print(f"Final airspeed:          {data['airspeed'][-1]:.3f} m/s")
    print(f"Final roll angle:        {np.rad2deg(data['phi'][-1]):.3f} deg")
    print(f"Final pitch angle:       {np.rad2deg(data['theta'][-1]):.3f} deg")
    print(f"Final heading angle:     {np.rad2deg(data['psi'][-1]):.3f} deg")
    print(f"Final altitude error:    {altitude_error_final:.3f} m")
    print(f"Final airspeed error:    {airspeed_error_final:.3f} m/s")
    print(f"Final heading error:     {heading_error_final_deg:.3f} deg")
    print(f"Max abs roll angle:      {np.max(np.abs(np.rad2deg(data['phi']))):.3f} deg")
    print(f"Max abs pitch angle:     {np.max(np.abs(np.rad2deg(data['theta']))):.3f} deg")
    print(f"Max abs sideslip:        {np.max(np.abs(np.rad2deg(data['beta']))):.3f} deg")
    print(f"Max abs elevator:        {np.max(np.abs(np.rad2deg(data['delta_e']))):.3f} deg")
    print(f"Max abs aileron:         {np.max(np.abs(np.rad2deg(data['delta_a']))):.3f} deg")
    print(f"Max abs rudder:          {np.max(np.abs(np.rad2deg(data['delta_r']))):.3f} deg")
    print(f"Throttle range:          {np.min(data['delta_t']):.3f} to {np.max(data['delta_t']):.3f}")


def plot_trajectory(data, config):
    """
    Plots closed-loop 3D trajectory.
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        data["x"],
        data["y"],
        data["altitude"],
        label="Aircraft trajectory",
    )

    ax.set_title("6DOF Closed-Loop Trajectory")
    ax.set_xlabel("x / North [m]")
    ax.set_ylabel("y / East [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend()

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_trajectory_3d.png",
    )


def plot_closed_loop_response(t, data, config):
    """
    Plots closed-loop tracking and aircraft response.
    """

    fig, axs = plt.subplots(5, 2, figsize=(15, 16))
    fig.suptitle("6DOF Closed-Loop Aircraft Response", fontsize=14)

    axs[0, 0].plot(t, data["altitude"], label="Actual Altitude")
    axs[0, 0].plot(t, data["altitude_command"], "--", label="Commanded Altitude")
    axs[0, 0].set_title("Altitude Tracking")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Altitude [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, data["airspeed"], label="Actual Airspeed")
    axs[0, 1].plot(t, data["airspeed_command"], "--", label="Commanded Airspeed")
    axs[0, 1].set_title("Airspeed Tracking")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Airspeed [m/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(data["psi"]), label="Actual Heading")
    axs[1, 0].plot(t, np.rad2deg(data["heading_command"]), "--", label="Commanded Heading")
    axs[1, 0].set_title("Heading Tracking")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Heading [deg]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(data["phi"]), label="Actual Roll")
    axs[1, 1].plot(t, np.rad2deg(data["phi_command"]), "--", label="Commanded Roll")
    axs[1, 1].set_title("Roll Tracking")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Roll [deg]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, np.rad2deg(data["theta"]), label="Actual Pitch")
    axs[2, 0].plot(t, np.rad2deg(data["theta_command"]), "--", label="Commanded Pitch")
    axs[2, 0].set_title("Pitch Tracking")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Pitch [deg]")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].plot(t, np.rad2deg(data["alpha"]), label="AoA α")
    axs[2, 1].plot(t, np.rad2deg(data["beta"]), label="Sideslip β")
    axs[2, 1].set_title("Air Data Angles")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Angle [deg]")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    axs[3, 0].plot(t, np.rad2deg(data["p_rate"]), label="p")
    axs[3, 0].plot(t, np.rad2deg(data["q_rate"]), label="q")
    axs[3, 0].plot(t, np.rad2deg(data["r_rate"]), label="r")
    axs[3, 0].set_title("Body Rates")
    axs[3, 0].set_xlabel("Time [s]")
    axs[3, 0].set_ylabel("Rate [deg/s]")
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    axs[3, 1].plot(t, data["u"], label="u")
    axs[3, 1].plot(t, data["v"], label="v")
    axs[3, 1].plot(t, data["w"], label="w")
    axs[3, 1].set_title("Body Velocities")
    axs[3, 1].set_xlabel("Time [s]")
    axs[3, 1].set_ylabel("Velocity [m/s]")
    axs[3, 1].legend()
    axs[3, 1].grid(True)

    axs[4, 0].plot(t, np.rad2deg(data["delta_e"]), label="Elevator")
    axs[4, 0].plot(t, np.rad2deg(data["delta_a"]), label="Aileron")
    axs[4, 0].plot(t, np.rad2deg(data["delta_r"]), label="Rudder")
    axs[4, 0].set_title("Control Surface Commands")
    axs[4, 0].set_xlabel("Time [s]")
    axs[4, 0].set_ylabel("Deflection [deg]")
    axs[4, 0].legend()
    axs[4, 0].grid(True)

    axs[4, 1].plot(t, data["delta_t"], label="Throttle")
    axs[4, 1].set_title("Throttle Command")
    axs[4, 1].set_xlabel("Time [s]")
    axs[4, 1].set_ylabel("Throttle [0–1]")
    axs[4, 1].legend()
    axs[4, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_closed_loop_response.png",
    )


def run_closed_loop_6dof(
    config: ClosedLoop6DOFScenario,
    gains: Controller6DOFGains = DEFAULT_6DOF_GAINS,
    show_plots=True,
):
    """
    Runs the closed-loop 6DOF autopilot simulation.

    Returns:
        t
        sol
        data
    """

    print("\nClosed-loop 6DOF scenario:")
    print("--------------------------")
    print(f"Scenario name:      {config.scenario_name}")
    print(f"Target airspeed:    {config.target_airspeed_mps:.3f} m/s")
    print(f"Trim altitude:      {config.trim_altitude_m:.3f} m")
    print(f"Altitude step:      {config.altitude_step_m:.3f} m")
    print(f"Heading step:       {config.heading_step_deg:.3f} deg")
    print(f"Airspeed step:      {config.airspeed_step_mps:.3f} m/s")
    print(f"Simulation time:    {config.sim_start_time_s:.3f} to {config.sim_end_time_s:.3f} s")

    trim_state, trim_controls, trim_info = solve_trim_6dof(
        target_airspeed_mps=config.target_airspeed_mps,
        altitude_m=config.trim_altitude_m,
    )

    print_trim_summary(trim_info)

    controls_function = build_closed_loop_controls_function(
        trim_controls=trim_controls,
        trim_info=trim_info,
        config=config,
        gains=gains,
    )

    def dynamics_wrapper(t, state):
        controls = controls_function(t, state)

        dynamics_controls = {
            "delta_e": controls["delta_e"],
            "delta_a": controls["delta_a"],
            "delta_r": controls["delta_r"],
            "delta_t": controls["delta_t"],
            "wind_inertial": controls["wind_inertial"],
        }

        return dynamics_6dof(
            t=t,
            state=state,
            controls=dynamics_controls,
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
        raise RuntimeError(f"Closed-loop 6DOF simulation failed: {sol.message}")

    t = sol.t

    data = post_process_solution(
        t=t,
        sol_y=sol.y,
        controls_function=controls_function,
    )

    print_simulation_summary(
        data=data,
        config=config,
    )

    plot_trajectory(
        data=data,
        config=config,
    )

    plot_closed_loop_response(
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
    Entry point for first closed-loop 6DOF simulation.
    """

    config = ClosedLoop6DOFScenario()
    gains = DEFAULT_6DOF_GAINS

    run_closed_loop_6dof(
        config=config,
        gains=gains,
        show_plots=True,
    )


if __name__ == "__main__":
    main()