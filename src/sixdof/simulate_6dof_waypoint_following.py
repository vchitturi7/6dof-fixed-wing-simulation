# simulate_6dof_waypoint_following.py
# 6DOF waypoint-following simulation using fixed-step RK4 integration.
#
# Architecture:
#
#   Waypoint course
#       -> line-segment path-following guidance
#       -> heading / altitude / airspeed commands
#       -> 6DOF autopilot
#       -> aircraft dynamics
#
# This is the final waypoint-following simulation file.
#
# It uses:
#   - final tuned 6DOF autopilot gains
#   - fixed-step RK4 integration for clean waypoint sequencing
#   - selectable waypoint courses from waypoint_course_config.py
#
# Usage from SRC folder:
#   python sixdof/simulate_6dof_waypoint_following.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trim_6dof import solve_trim_6dof
from dynamics_6dof import dynamics_6dof, wind_body_from_controls
from aerodynamics_6dof import compute_air_data
from controllers_6dof import (
    Controller6DOFGains,
    full_6dof_autopilot,
)
from waypoint_guidance import (
    WaypointGuidanceConfig,
    compute_waypoint_commands,
)
from waypoint_course_config import get_waypoint_course


FINAL_TUNED_6DOF_GAINS = Controller6DOFGains(
    # Longitudinal tuned gains
    pitch_kp=1.25,
    pitch_kd=0.50,
    altitude_kh_rad_per_m=0.0065,
    max_pitch_command_offset_rad=np.deg2rad(12.0),
    airspeed_kv=0.075,

    # Lateral tuned gains
    roll_kp=0.85,
    roll_kd=0.90,
    heading_kp_rad_per_rad=0.35,
    max_roll_command_rad=np.deg2rad(18.0),
    yaw_damper_kr=0.08,
    sideslip_kb=0.03,
)


class WaypointFollowingScenario:
    """
    Scenario configuration for 6DOF waypoint-following.
    """

    def __init__(self):
        self.scenario_name = "sixdof_waypoint_following"

        # Trim / initial condition
        self.target_airspeed_mps = 25.0
        self.trim_altitude_m = 100.0

        # Simulation time
        self.sim_start_time_s = 0.0
        self.sim_end_time_s = 140.0
        self.time_step_s = 0.02

        # Wind in inertial NED frame
        self.wind_x_inertial_mps = 0.0
        self.wind_y_inertial_mps = 0.0
        self.wind_z_down_inertial_mps = 0.0

        # Waypoint course selection
        #
        # Valid course names from waypoint_course_config.py:
        #   "default"
        #   "easy"
        #   "aggressive"
        #   "straight_climb"
        self.course_name = "default"

        # Guidance parameters
        self.waypoint_capture_radius_m = 55.0
        self.max_heading_change_per_command_deg = 45.0

        # Output
        self.save_plots = True
        self.results_dir = "results/6dof/waypoint_following"


def save_figure(fig, config, filename):
    """
    Saves a matplotlib figure if config.save_plots is True.
    """

    if not config.save_plots:
        return

    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved figure: {output_path}")


def get_wind_inertial(config):
    """
    Returns inertial-frame wind vector.
    """

    return np.array([
        config.wind_x_inertial_mps,
        config.wind_y_inertial_mps,
        config.wind_z_down_inertial_mps,
    ])


def compute_air_data_from_state(state, controls):
    """
    Computes air data from aircraft state and wind.
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

    return compute_air_data(
        velocity_body=velocity_body,
        wind_body=wind_body,
    )


def build_controls_for_state(
    t,
    state,
    trim_controls,
    trim_info,
    gains,
    waypoints,
    guidance_config,
    current_waypoint_index,
    wind_inertial,
):
    """
    Computes guidance commands, autopilot controls, and guidance info
    for one aircraft state.
    """

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

    commands, guidance_info = compute_waypoint_commands(
        state=state,
        waypoints=waypoints,
        current_waypoint_index=current_waypoint_index,
        config=guidance_config,
    )

    autopilot_controls = full_6dof_autopilot(
        commands=commands,
        state=state,
        air_data=air_data,
        trim_controls=trim_controls,
        trim_info=trim_info,
        gains=gains,
    )

    controls = {
        "delta_e": autopilot_controls["delta_e"],
        "delta_a": autopilot_controls["delta_a"],
        "delta_r": autopilot_controls["delta_r"],
        "delta_t": autopilot_controls["delta_t"],
        "wind_inertial": wind_inertial,
    }

    record = {
        "delta_e": autopilot_controls["delta_e"],
        "delta_a": autopilot_controls["delta_a"],
        "delta_r": autopilot_controls["delta_r"],
        "delta_t": autopilot_controls["delta_t"],
        "theta_command_rad": autopilot_controls["theta_command_rad"],
        "phi_command_rad": autopilot_controls["phi_command_rad"],
        "altitude_command_m": commands["altitude_command_m"],
        "airspeed_command_mps": commands["airspeed_command_mps"],
        "heading_command_rad": commands["heading_command_rad"],
        "active_waypoint_index": guidance_info["active_waypoint_index"],
        "distance_to_waypoint_m": guidance_info["distance_to_waypoint_m"],
        "desired_heading_rad": guidance_info["desired_heading_rad"],
        "path_heading_rad": guidance_info["path_heading_rad"],
        "cross_track_error_m": guidance_info["cross_track_error_m"],
        "along_track_distance_m": guidance_info["along_track_distance_m"],
        "leg_length_m": guidance_info["leg_length_m"],
        "progress_fraction": guidance_info["progress_fraction"],
    }

    return controls, record, air_data, guidance_info


def rk4_step(dynamics_function, t, state, dt):
    """
    One fixed-step RK4 integration step.
    """

    k1 = dynamics_function(t, state)
    k2 = dynamics_function(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = dynamics_function(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = dynamics_function(t + dt, state + dt * k3)

    next_state = state + (dt / 6.0) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )

    return next_state


def run_fixed_step_waypoint_simulation(
    initial_state,
    trim_controls,
    trim_info,
    gains,
    scenario,
    waypoints,
    guidance_config,
):
    """
    Runs waypoint-following using fixed-step RK4.

    Returns:
        t_array
        state_history
        record_history

    Why fixed-step?
        Waypoint index is discrete mission logic.
        Adaptive ODE solvers can call the dynamics function at internal
        non-output times, which can make waypoint sequencing fragile.
        Fixed-step RK4 keeps waypoint updates clean and monotonic.
    """

    wind_inertial = get_wind_inertial(scenario)

    t_array = np.arange(
        scenario.sim_start_time_s,
        scenario.sim_end_time_s + scenario.time_step_s,
        scenario.time_step_s,
    )

    number_of_steps = len(t_array)

    state_history = np.zeros((12, number_of_steps))
    state_history[:, 0] = initial_state

    current_waypoint_index = 0

    record_history = []

    for i in range(number_of_steps - 1):
        t = t_array[i]
        state = state_history[:, i]

        controls, record, air_data, guidance_info = build_controls_for_state(
            t=t,
            state=state,
            trim_controls=trim_controls,
            trim_info=trim_info,
            gains=gains,
            waypoints=waypoints,
            guidance_config=guidance_config,
            current_waypoint_index=current_waypoint_index,
            wind_inertial=wind_inertial,
        )

        current_waypoint_index = guidance_info["active_waypoint_index"]

        record_history.append(record)

        def dynamics_for_rk4(t_local, state_local):
            # Hold controls constant over one small timestep.
            return dynamics_6dof(
                t=t_local,
                state=state_local,
                controls=controls,
            )

        next_state = rk4_step(
            dynamics_function=dynamics_for_rk4,
            t=t,
            state=state,
            dt=scenario.time_step_s,
        )

        state_history[:, i + 1] = next_state

    # Record final sample.
    final_t = t_array[-1]
    final_state = state_history[:, -1]

    controls, record, air_data, guidance_info = build_controls_for_state(
        t=final_t,
        state=final_state,
        trim_controls=trim_controls,
        trim_info=trim_info,
        gains=gains,
        waypoints=waypoints,
        guidance_config=guidance_config,
        current_waypoint_index=current_waypoint_index,
        wind_inertial=wind_inertial,
    )

    record_history.append(record)

    return t_array, state_history, record_history


def post_process_solution(t, state_history, record_history, scenario):
    """
    Converts state, control, and guidance histories into plotting arrays.
    """

    x = state_history[0]
    y = state_history[1]
    z_down = state_history[2]

    u = state_history[3]
    v = state_history[4]
    w = state_history[5]

    phi = state_history[6]
    theta = state_history[7]
    psi = state_history[8]

    p_rate = state_history[9]
    q_rate = state_history[10]
    r_rate = state_history[11]

    altitude = -z_down

    wind_inertial = get_wind_inertial(scenario)

    airspeed = []
    alpha = []
    beta = []

    for i in range(len(t)):
        state_i = state_history[:, i]

        controls_for_air_data = {
            "delta_e": 0.0,
            "delta_a": 0.0,
            "delta_r": 0.0,
            "delta_t": 0.0,
            "wind_inertial": wind_inertial,
        }

        air_data_i = compute_air_data_from_state(
            state=state_i,
            controls=controls_for_air_data,
        )

        airspeed.append(air_data_i["V"])
        alpha.append(air_data_i["alpha"])
        beta.append(air_data_i["beta"])

    delta_e = np.array([
        record["delta_e"]
        for record in record_history
    ])

    delta_a = np.array([
        record["delta_a"]
        for record in record_history
    ])

    delta_r = np.array([
        record["delta_r"]
        for record in record_history
    ])

    delta_t = np.array([
        record["delta_t"]
        for record in record_history
    ])

    theta_command = np.array([
        record["theta_command_rad"]
        for record in record_history
    ])

    phi_command = np.array([
        record["phi_command_rad"]
        for record in record_history
    ])

    altitude_command = np.array([
        record["altitude_command_m"]
        for record in record_history
    ])

    airspeed_command = np.array([
        record["airspeed_command_mps"]
        for record in record_history
    ])

    heading_command = np.array([
        record["heading_command_rad"]
        for record in record_history
    ])

    active_waypoint_index = np.array([
        record["active_waypoint_index"]
        for record in record_history
    ])

    distance_to_waypoint = np.array([
        record["distance_to_waypoint_m"]
        for record in record_history
    ])

    desired_heading = np.array([
        record["desired_heading_rad"]
        for record in record_history
    ])

    path_heading = np.array([
        record["path_heading_rad"]
        for record in record_history
    ])

    cross_track_error = np.array([
        record["cross_track_error_m"]
        for record in record_history
    ])

    along_track_distance = np.array([
        record["along_track_distance_m"]
        for record in record_history
    ])

    leg_length = np.array([
        record["leg_length_m"]
        for record in record_history
    ])

    progress_fraction = np.array([
        record["progress_fraction"]
        for record in record_history
    ])

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
        "delta_e": delta_e,
        "delta_a": delta_a,
        "delta_r": delta_r,
        "delta_t": delta_t,
        "theta_command": theta_command,
        "phi_command": phi_command,
        "altitude_command": altitude_command,
        "airspeed_command": airspeed_command,
        "heading_command": heading_command,
        "desired_heading": desired_heading,
        "path_heading": path_heading,
        "active_waypoint_index": active_waypoint_index,
        "distance_to_waypoint": distance_to_waypoint,
        "cross_track_error": cross_track_error,
        "along_track_distance": along_track_distance,
        "leg_length": leg_length,
        "progress_fraction": progress_fraction,
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


def print_waypoint_summary(data, waypoints, scenario):
    """
    Prints final waypoint-following summary.
    """

    final_waypoint_index = int(data["active_waypoint_index"][-1])
    final_distance = data["distance_to_waypoint"][-1]

    print("\nWaypoint-following summary:")
    print("---------------------------")
    print(f"Course name:               {scenario.course_name}")
    print(f"Final x position:          {data['x'][-1]:.3f} m")
    print(f"Final y position:          {data['y'][-1]:.3f} m")
    print(f"Final altitude:            {data['altitude'][-1]:.3f} m")
    print(f"Final airspeed:            {data['airspeed'][-1]:.3f} m/s")
    print(f"Final heading:             {np.rad2deg(data['psi'][-1]):.3f} deg")
    print(f"Final waypoint index:      {final_waypoint_index}")
    print(f"Final waypoint distance:   {final_distance:.3f} m")
    print(f"Total waypoints:           {len(waypoints)}")
    print(f"Max abs roll:              {np.max(np.abs(np.rad2deg(data['phi']))):.3f} deg")
    print(f"Max abs pitch:             {np.max(np.abs(np.rad2deg(data['theta']))):.3f} deg")
    print(f"Max abs sideslip:          {np.max(np.abs(np.rad2deg(data['beta']))):.3f} deg")
    print(f"Max abs aileron:           {np.max(np.abs(np.rad2deg(data['delta_a']))):.3f} deg")
    print(f"Max abs elevator:          {np.max(np.abs(np.rad2deg(data['delta_e']))):.3f} deg")
    print(f"Max abs rudder:            {np.max(np.abs(np.rad2deg(data['delta_r']))):.3f} deg")
    print(f"Throttle range:            {np.min(data['delta_t']):.3f} to {np.max(data['delta_t']):.3f}")
    print(f"Max abs cross-track error: {np.max(np.abs(data['cross_track_error'])):.3f} m")


def plot_top_down_path(data, waypoints, scenario):
    """
    Plots top-down x-y path with waypoints.
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        data["x"],
        data["y"],
        label="Aircraft path",
    )

    waypoint_x = [
        waypoint.x_m
        for waypoint in waypoints
    ]

    waypoint_y = [
        waypoint.y_m
        for waypoint in waypoints
    ]

    ax.scatter(
        waypoint_x,
        waypoint_y,
        marker="x",
        s=80,
        label="Waypoints",
    )

    for i, waypoint in enumerate(waypoints):
        ax.annotate(
            f"WP{i}",
            xy=(waypoint.x_m, waypoint.y_m),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_title(f"6DOF Waypoint-Following Top-Down Path ({scenario.course_name})")
    ax.set_xlabel("x / North [m]")
    ax.set_ylabel("y / East [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    save_figure(
        fig=fig,
        config=scenario,
        filename=f"{scenario.scenario_name}_{scenario.course_name}_top_down_path.png",
    )


def plot_3d_path(data, waypoints, scenario):
    """
    Plots 3D trajectory with waypoints.
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        data["x"],
        data["y"],
        data["altitude"],
        label="Aircraft path",
    )

    waypoint_x = [
        waypoint.x_m
        for waypoint in waypoints
    ]

    waypoint_y = [
        waypoint.y_m
        for waypoint in waypoints
    ]

    waypoint_altitude = [
        waypoint.altitude_m
        for waypoint in waypoints
    ]

    ax.scatter(
        waypoint_x,
        waypoint_y,
        waypoint_altitude,
        marker="x",
        s=80,
        label="Waypoints",
    )

    ax.set_title(f"6DOF Waypoint-Following 3D Path ({scenario.course_name})")
    ax.set_xlabel("x / North [m]")
    ax.set_ylabel("y / East [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend()

    save_figure(
        fig=fig,
        config=scenario,
        filename=f"{scenario.scenario_name}_{scenario.course_name}_3d_path.png",
    )


def plot_waypoint_response(t, data, scenario):
    """
    Plots waypoint-following response histories.
    """

    fig, axs = plt.subplots(5, 2, figsize=(15, 16))
    fig.suptitle(
        f"6DOF Waypoint-Following Response ({scenario.course_name})",
        fontsize=14,
    )

    axs[0, 0].plot(t, data["altitude"], label="Actual altitude")
    axs[0, 0].plot(t, data["altitude_command"], "--", label="Commanded altitude")
    axs[0, 0].set_title("Altitude")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Altitude [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, data["airspeed"], label="Actual airspeed")
    axs[0, 1].plot(t, data["airspeed_command"], "--", label="Commanded airspeed")
    axs[0, 1].set_title("Airspeed")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Airspeed [m/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(data["psi"]), label="Heading")
    axs[1, 0].plot(t, np.rad2deg(data["heading_command"]), "--", label="Heading command")
    axs[1, 0].plot(t, np.rad2deg(data["path_heading"]), ":", label="Path heading")
    axs[1, 0].set_title("Heading")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Heading [deg]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(data["phi"]), label="Roll")
    axs[1, 1].plot(t, np.rad2deg(data["phi_command"]), "--", label="Roll command")
    axs[1, 1].set_title("Roll")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Roll [deg]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, np.rad2deg(data["theta"]), label="Pitch")
    axs[2, 0].plot(t, np.rad2deg(data["theta_command"]), "--", label="Pitch command")
    axs[2, 0].set_title("Pitch")
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

    axs[3, 0].plot(t, data["active_waypoint_index"])
    axs[3, 0].set_title("Active Waypoint Index")
    axs[3, 0].set_xlabel("Time [s]")
    axs[3, 0].set_ylabel("Waypoint index")
    axs[3, 0].grid(True)

    axs[3, 1].plot(t, data["distance_to_waypoint"])
    axs[3, 1].set_title("Distance to Active Waypoint")
    axs[3, 1].set_xlabel("Time [s]")
    axs[3, 1].set_ylabel("Distance [m]")
    axs[3, 1].grid(True)

    axs[4, 0].plot(t, data["cross_track_error"])
    axs[4, 0].set_title("Cross-Track Error")
    axs[4, 0].set_xlabel("Time [s]")
    axs[4, 0].set_ylabel("Cross-track error [m]")
    axs[4, 0].grid(True)

    axs[4, 1].plot(t, np.rad2deg(data["delta_e"]), label="Elevator")
    axs[4, 1].plot(t, np.rad2deg(data["delta_a"]), label="Aileron")
    axs[4, 1].plot(t, np.rad2deg(data["delta_r"]), label="Rudder")
    axs[4, 1].set_title("Control Surface Commands")
    axs[4, 1].set_xlabel("Time [s]")
    axs[4, 1].set_ylabel("Deflection [deg]")
    axs[4, 1].legend()
    axs[4, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(
        fig=fig,
        config=scenario,
        filename=f"{scenario.scenario_name}_{scenario.course_name}_response.png",
    )


def run_waypoint_following_6dof(
    scenario: WaypointFollowingScenario,
    gains: Controller6DOFGains = FINAL_TUNED_6DOF_GAINS,
    show_plots=True,
):
    """
    Runs the waypoint-following simulation.
    """

    print("\n6DOF waypoint-following scenario:")
    print("---------------------------------")
    print(f"Scenario name:      {scenario.scenario_name}")
    print(f"Course name:        {scenario.course_name}")
    print(f"Target airspeed:    {scenario.target_airspeed_mps:.3f} m/s")
    print(f"Trim altitude:      {scenario.trim_altitude_m:.3f} m")
    print(f"Simulation time:    {scenario.sim_start_time_s:.3f} to {scenario.sim_end_time_s:.3f} s")
    print(f"Time step:          {scenario.time_step_s:.4f} s")

    trim_state, trim_controls, trim_info = solve_trim_6dof(
        target_airspeed_mps=scenario.target_airspeed_mps,
        altitude_m=scenario.trim_altitude_m,
    )

    print_trim_summary(trim_info)

    waypoints = get_waypoint_course(
        scenario.course_name
    )

    guidance_config = WaypointGuidanceConfig(
        waypoint_capture_radius_m=scenario.waypoint_capture_radius_m,
        max_heading_change_per_command_rad=np.deg2rad(
            scenario.max_heading_change_per_command_deg
        ),
    )

    t, state_history, record_history = run_fixed_step_waypoint_simulation(
        initial_state=trim_state,
        trim_controls=trim_controls,
        trim_info=trim_info,
        gains=gains,
        scenario=scenario,
        waypoints=waypoints,
        guidance_config=guidance_config,
    )

    data = post_process_solution(
        t=t,
        state_history=state_history,
        record_history=record_history,
        scenario=scenario,
    )

    print_waypoint_summary(
        data=data,
        waypoints=waypoints,
        scenario=scenario,
    )

    plot_top_down_path(
        data=data,
        waypoints=waypoints,
        scenario=scenario,
    )

    plot_3d_path(
        data=data,
        waypoints=waypoints,
        scenario=scenario,
    )

    plot_waypoint_response(
        t=t,
        data=data,
        scenario=scenario,
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return t, state_history, data


def main():
    """
    Entry point for waypoint-following sim.

    To test another course, change:
        scenario.course_name = "default"

    Options:
        "default"
        "easy"
        "aggressive"
        "straight_climb"
    """

    scenario = WaypointFollowingScenario()

    # Change this line to test different waypoint courses.
    scenario.course_name = "aggressive"

    run_waypoint_following_6dof(
        scenario=scenario,
        gains=FINAL_TUNED_6DOF_GAINS,
        show_plots=True,
    )


if __name__ == "__main__":
    main()