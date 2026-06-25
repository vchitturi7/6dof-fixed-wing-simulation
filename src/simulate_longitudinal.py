# simulate_longitudinal.py
# Runs a trimmed longitudinal simulation with altitude hold, airspeed hold,
# pitch PID, configurable inertial-frame gust disturbance, actuator dynamics,
# anti-windup, performance metrics, separated plotting figures, JSON run logging,
# and configurable gains.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import aircraft_params as p
from longitudinal_dynamics import longitudinal_dynamics, inertial_wind_to_body
from trim_longitudinal import solve_trim
from scenario_config import DEFAULT_SCENARIO, ScenarioConfig
from controller_config import DEFAULT_GAINS, TUNED_GAINS, ControllerGains
from run_logging import save_run_config
from controllers import (
    pitch_hold_pid_controller,
    altitude_hold_controller,
    airspeed_hold_controller,
)


def smooth_step_0_to_1(x):
    """
    Smoothly transitions from 0 to 1 as x goes from 0 to 1.

    Uses a half-cosine profile:
        x = 0 -> 0
        x = 1 -> 1
    """

    x_clipped = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x_clipped))


def gust_profile(t, config):
    """
    Defines inertial-frame wind/gust inputs from the scenario config.

    Inertial-frame convention:
        +wind_x_inertial = wind moving forward/horizontally
        +wind_z_inertial = upward vertical wind

    If config.use_smooth_gust is True:
        The gust ramps smoothly up and down.

    If config.use_smooth_gust is False:
        The gust is a square pulse.
    """

    if not (config.gust_start_time_s <= t <= config.gust_end_time_s):
        return 0.0, 0.0

    if not config.use_smooth_gust:
        return config.gust_wind_x_mps, config.gust_wind_z_mps

    gust_duration = config.gust_end_time_s - config.gust_start_time_s
    gust_mid_time = 0.5 * (config.gust_start_time_s + config.gust_end_time_s)

    if gust_duration <= 0.0:
        return 0.0, 0.0

    # Ramp up during first half, ramp down during second half.
    if t <= gust_mid_time:
        ramp_fraction = (t - config.gust_start_time_s) / (0.5 * gust_duration)
        scale = smooth_step_0_to_1(ramp_fraction)
    else:
        ramp_fraction = (t - gust_mid_time) / (0.5 * gust_duration)
        scale = 1.0 - smooth_step_0_to_1(ramp_fraction)

    wind_x_inertial = scale * config.gust_wind_x_mps
    wind_z_inertial = scale * config.gust_wind_z_mps

    return wind_x_inertial, wind_z_inertial


def compute_relative_airflow(u, w, theta, wind_x_inertial, wind_z_inertial):
    """
    Computes body-frame relative airflow quantities.

    Inputs:
        u, w              = body-frame aircraft velocity components, m/s
        theta             = pitch angle, rad
        wind_x_inertial   = inertial-frame horizontal wind, m/s
        wind_z_inertial   = inertial-frame vertical wind, m/s

    Returns:
        v_air             = relative airspeed, m/s
        alpha             = angle of attack, rad
        u_wind_body       = body x wind component, m/s
        w_wind_body       = body z wind component, m/s
    """

    u_wind_body, w_wind_body = inertial_wind_to_body(
        wind_x_inertial=wind_x_inertial,
        wind_z_inertial=wind_z_inertial,
        theta=theta,
    )

    u_rel = u - u_wind_body
    w_rel = w - w_wind_body

    v_air = np.sqrt(u_rel**2 + w_rel**2)
    alpha = np.arctan2(w_rel, u_rel)

    return v_air, alpha, u_wind_body, w_wind_body


def make_controls(trim_controls, trim_info, config, gains):
    """
    Control architecture:

        Altitude command -> theta command -> pitch PID -> elevator command
        Airspeed command -> throttle controller -> throttle command

    Important:
        The airspeed controller uses relative airspeed, not ground-relative speed.
    """

    trim_elevator = trim_controls["delta_e"]
    trim_throttle = trim_controls["delta_t"]
    theta_trim = trim_info["theta_trim_rad"]

    h_trim = config.trim_altitude_m
    v_trim = trim_info["target_airspeed"]

    def controls(t, full_state):
        u, w, q, theta, h, theta_error_integral = full_state[:6]

        wind_x_inertial, wind_z_inertial = gust_profile(t, config)

        v_air, _, _, _ = compute_relative_airflow(
            u=u,
            w=w,
            theta=theta,
            wind_x_inertial=wind_x_inertial,
            wind_z_inertial=wind_z_inertial,
        )

        if t < config.command_start_time_s:
            h_command = h_trim
            v_command = v_trim
        else:
            h_command = h_trim + config.altitude_step_m
            v_command = v_trim

        theta_command = altitude_hold_controller(
            h_command=h_command,
            h=h,
            theta_trim=theta_trim,
            gains=gains,
        )

        delta_e_command = pitch_hold_pid_controller(
            theta_command=theta_command,
            theta=theta,
            q=q,
            theta_error_integral=theta_error_integral,
            trim_elevator=trim_elevator,
            gains=gains,
        )

        delta_t_command = airspeed_hold_controller(
            v_command=v_command,
            v=v_air,
            trim_throttle=trim_throttle,
            gains=gains,
        )

        return {
            "delta_e": delta_e_command,
            "delta_t": delta_t_command,
            "theta_command": theta_command,
            "h_command": h_command,
            "v_command": v_command,
            "wind_x_inertial": wind_x_inertial,
            "wind_z_inertial": wind_z_inertial,
        }

    return controls


def should_freeze_pitch_integral(
    theta_error,
    theta_error_integral,
    delta_e_command,
    gains,
):
    """
    Anti-windup logic for the pitch integral state.

    We freeze the integral when:
        1. The integral is already at its positive/negative clamp and the error
           would push it farther into the clamp.
        2. The elevator command is saturated and the current pitch error would
           demand even more elevator in the same saturated direction.

    Sign convention:
        Positive pitch error means aircraft is below commanded pitch.
        In this aircraft model, nose-up correction requires more negative elevator.
    """

    tolerance = 1e-6

    integral_upper_limit = gains.max_pitch_integral_rad_s
    integral_lower_limit = -gains.max_pitch_integral_rad_s

    # Integral state clamp logic.
    integral_at_upper_and_growing = (
        theta_error_integral >= integral_upper_limit
        and theta_error > 0.0
    )

    integral_at_lower_and_growing = (
        theta_error_integral <= integral_lower_limit
        and theta_error < 0.0
    )

    # Elevator saturation logic.
    # Positive pitch error wants more negative elevator.
    elevator_at_negative_limit_and_error_wants_more_negative = (
        delta_e_command <= -gains.max_elevator_rad + tolerance
        and theta_error > 0.0
    )

    # Negative pitch error wants more positive elevator.
    elevator_at_positive_limit_and_error_wants_more_positive = (
        delta_e_command >= gains.max_elevator_rad - tolerance
        and theta_error < 0.0
    )

    freeze_integral = (
        integral_at_upper_and_growing
        or integral_at_lower_and_growing
        or elevator_at_negative_limit_and_error_wants_more_negative
        or elevator_at_positive_limit_and_error_wants_more_positive
    )

    return freeze_integral


def compute_settling_time(t, error, tolerance, start_time):
    """
    Computes approximate settling time.

    Settling time is defined as the first time after start_time
    when abs(error) stays within tolerance for the rest of the simulation.

    Returns:
        settling_time, or None if it never settles.
    """

    start_index = np.searchsorted(t, start_time)

    for i in range(start_index, len(t)):
        if np.all(np.abs(error[i:]) <= tolerance):
            return t[i] - start_time

    return None


def print_scenario_summary(config, gains):
    """
    Prints the scenario setup and controller gains.
    """

    print("\nScenario configuration:")
    print("-----------------------")
    print(f"Scenario name:         {config.scenario_name}")
    print(f"Target airspeed:       {config.target_airspeed_mps:.3f} m/s")
    print(f"Trim altitude:         {config.trim_altitude_m:.3f} m")
    print(f"Altitude command step: {config.altitude_step_m:.3f} m")
    print(f"Command starts at:     {config.command_start_time_s:.3f} s")
    print(f"Simulation time:       {config.sim_start_time_s:.3f} to {config.sim_end_time_s:.3f} s")
    print(f"Gust window:           {config.gust_start_time_s:.3f} to {config.gust_end_time_s:.3f} s")
    print(f"Gust wind x:           {config.gust_wind_x_mps:.3f} m/s")
    print(f"Gust wind z:           {config.gust_wind_z_mps:.3f} m/s")
    print(f"Smooth gust:           {config.use_smooth_gust}")

    print("\nController gains:")
    print("-----------------")
    print(f"Pitch Kp:             {gains.pitch_kp:.4f}")
    print(f"Pitch Ki:             {gains.pitch_ki:.4f}")
    print(f"Pitch Kd:             {gains.pitch_kd:.4f}")
    print(f"Altitude Kh:          {gains.altitude_kh_rad_per_m:.6f} rad/m")
    print(f"Airspeed Kv:          {gains.airspeed_kv:.4f}")


def compute_performance_metrics(
    t,
    h,
    h_command,
    V_air,
    v_command,
    theta,
    theta_command,
    elevator_command,
    elevator_actual,
    throttle_command,
    throttle_actual,
    theta_error_integral,
    config,
    gains,
):
    """
    Computes performance metrics and returns them as a dictionary.
    """

    altitude_error = h_command - h
    airspeed_error = v_command - V_air
    pitch_error = theta_command - theta

    command_start_time = config.command_start_time_s
    gust_start_time = config.gust_start_time_s
    gust_end_time = config.gust_end_time_s

    after_command = t >= command_start_time
    during_gust = (t >= gust_start_time) & (t <= gust_end_time)
    after_gust = t > gust_end_time

    altitude_settling_time = compute_settling_time(
        t=t,
        error=altitude_error,
        tolerance=config.altitude_settling_tolerance_m,
        start_time=command_start_time,
    )

    airspeed_settling_time = compute_settling_time(
        t=t,
        error=airspeed_error,
        tolerance=config.airspeed_settling_tolerance_mps,
        start_time=command_start_time,
    )

    if np.any(during_gust):
        max_altitude_error_during_gust = np.max(np.abs(altitude_error[during_gust]))
        max_airspeed_error_during_gust = np.max(np.abs(airspeed_error[during_gust]))
    else:
        max_altitude_error_during_gust = np.nan
        max_airspeed_error_during_gust = np.nan

    if np.any(after_gust):
        max_altitude_error_after_gust = np.max(np.abs(altitude_error[after_gust]))
        max_airspeed_error_after_gust = np.max(np.abs(airspeed_error[after_gust]))
    else:
        max_altitude_error_after_gust = np.nan
        max_airspeed_error_after_gust = np.nan

    metrics = {
        "scenario_name": config.scenario_name,
        "target_airspeed_mps": config.target_airspeed_mps,
        "trim_altitude_m": config.trim_altitude_m,
        "altitude_step_m": config.altitude_step_m,
        "gust_wind_x_mps": config.gust_wind_x_mps,
        "gust_wind_z_mps": config.gust_wind_z_mps,
        "use_smooth_gust": config.use_smooth_gust,
        "pitch_kp": gains.pitch_kp,
        "pitch_ki": gains.pitch_ki,
        "pitch_kd": gains.pitch_kd,
        "altitude_kh_rad_per_m": gains.altitude_kh_rad_per_m,
        "airspeed_kv": gains.airspeed_kv,
        "max_altitude_error_after_command_m": np.max(np.abs(altitude_error[after_command])),
        "final_altitude_error_m": altitude_error[-1],
        "altitude_settling_time_s": altitude_settling_time,
        "max_airspeed_error_after_command_mps": np.max(np.abs(airspeed_error[after_command])),
        "final_airspeed_error_mps": airspeed_error[-1],
        "airspeed_settling_time_s": airspeed_settling_time,
        "max_pitch_error_after_command_deg": np.max(np.abs(np.rad2deg(pitch_error[after_command]))),
        "final_pitch_error_deg": np.rad2deg(pitch_error[-1]),
        "max_elevator_command_deg": np.max(np.abs(np.rad2deg(elevator_command))),
        "max_elevator_actual_deg": np.max(np.abs(np.rad2deg(elevator_actual))),
        "max_throttle_command": np.max(throttle_command),
        "max_throttle_actual": np.max(throttle_actual),
        "max_pitch_integral_deg_s": np.max(np.abs(np.rad2deg(theta_error_integral))),
        "max_altitude_error_during_gust_m": max_altitude_error_during_gust,
        "max_airspeed_error_during_gust_mps": max_airspeed_error_during_gust,
        "max_altitude_error_after_gust_m": max_altitude_error_after_gust,
        "max_airspeed_error_after_gust_mps": max_airspeed_error_after_gust,
    }

    return metrics


def print_performance_metrics(metrics, config):
    """
    Prints performance metrics from the metrics dictionary.
    """

    print("\nPerformance metrics:")
    print("--------------------")
    print(f"Max altitude error after command: {metrics['max_altitude_error_after_command_m']:.3f} m")
    print(f"Final altitude error:             {metrics['final_altitude_error_m']:.3f} m")

    if metrics["altitude_settling_time_s"] is None:
        print(
            f"Altitude settling time ±{config.altitude_settling_tolerance_m:.2f} m: "
            "Did not settle"
        )
    else:
        print(
            f"Altitude settling time ±{config.altitude_settling_tolerance_m:.2f} m: "
            f"{metrics['altitude_settling_time_s']:.3f} s"
        )

    print(f"Max airspeed error after command: {metrics['max_airspeed_error_after_command_mps']:.3f} m/s")
    print(f"Final airspeed error:             {metrics['final_airspeed_error_mps']:.3f} m/s")

    if metrics["airspeed_settling_time_s"] is None:
        print(
            f"Airspeed settling time ±{config.airspeed_settling_tolerance_mps:.2f} m/s: "
            "Did not settle"
        )
    else:
        print(
            f"Airspeed settling time ±{config.airspeed_settling_tolerance_mps:.2f} m/s: "
            f"{metrics['airspeed_settling_time_s']:.3f} s"
        )

    print(f"Max pitch error after command:    {metrics['max_pitch_error_after_command_deg']:.3f} deg")
    print(f"Final pitch error:                {metrics['final_pitch_error_deg']:.3f} deg")
    print(f"Max elevator command:             {metrics['max_elevator_command_deg']:.3f} deg")
    print(f"Max elevator actual:              {metrics['max_elevator_actual_deg']:.3f} deg")
    print(f"Max throttle command:             {metrics['max_throttle_command']:.3f}")
    print(f"Max throttle actual:              {metrics['max_throttle_actual']:.3f}")
    print(f"Max pitch integral:               {metrics['max_pitch_integral_deg_s']:.3f} deg*s")
    print(f"Max altitude error during gust:   {metrics['max_altitude_error_during_gust_m']:.3f} m")
    print(f"Max airspeed error during gust:   {metrics['max_airspeed_error_during_gust_mps']:.3f} m/s")
    print(f"Max altitude error after gust:    {metrics['max_altitude_error_after_gust_m']:.3f} m")
    print(f"Max airspeed error after gust:    {metrics['max_airspeed_error_after_gust_mps']:.3f} m/s")


def save_figure(fig, config, filename):
    """
    Saves a matplotlib figure into the configured results directory.
    """

    if not config.save_plots:
        return

    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    output_path = results_path / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def plot_flight_response(
    t,
    h,
    h_command,
    V_air,
    v_command,
    theta,
    theta_command,
    alpha,
    q,
    elevator_command,
    elevator_actual,
    throttle_command,
    throttle_actual,
    config,
):
    """
    Figure 1:
    Main aircraft response plots.
    """

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Flight Response: Altitude/Airspeed Hold with Gust", fontsize=14)

    axs[0, 0].plot(t, h, label="Actual Altitude")
    axs[0, 0].plot(t, h_command, "--", label="Commanded Altitude")
    axs[0, 0].set_title("Altitude Tracking")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("h [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, V_air, label="Actual Airspeed")
    axs[0, 1].plot(t, v_command, "--", label="Commanded Airspeed")
    axs[0, 1].set_title("Airspeed Tracking")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("V_air [m/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.rad2deg(theta), label="Actual Pitch")
    axs[1, 0].plot(t, np.rad2deg(theta_command), "--", label="Commanded Pitch")
    axs[1, 0].set_title("Pitch Tracking")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("θ [deg]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, np.rad2deg(alpha))
    axs[1, 1].set_title("Angle of Attack")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("α [deg]")
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, np.rad2deg(q))
    axs[2, 0].set_title("Pitch Rate")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("q [deg/s]")
    axs[2, 0].grid(True)

    axs[2, 1].plot(t, np.rad2deg(elevator_command), "--", label="Elevator Cmd [deg]", alpha=0.6)
    axs[2, 1].plot(t, np.rad2deg(elevator_actual), label="Elevator Actual [deg]")
    axs[2, 1].plot(t, throttle_command * 25.0, "--", label="Throttle Cmd × 25", alpha=0.6)
    axs[2, 1].plot(t, throttle_actual * 25.0, label="Throttle Actual × 25")
    axs[2, 1].set_title("Control Commands vs Actual")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Command")
    axs[2, 1].legend(fontsize=8)
    axs[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_flight_response.png",
    )


def plot_validation_and_disturbance(
    t,
    altitude_error,
    airspeed_error,
    pitch_error,
    theta_error_integral,
    wind_x_inertial,
    wind_z_inertial,
    u_wind_body,
    w_wind_body,
    config,
):
    """
    Figure 2:
    Disturbance, wind-frame conversion, and tracking error plots.
    """

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Validation: Disturbance Inputs and Tracking Errors", fontsize=14)

    axs[0, 0].plot(t, wind_x_inertial, label="Wind X Inertial")
    axs[0, 0].plot(t, wind_z_inertial, label="Wind Z Inertial")
    axs[0, 0].set_title("Inertial-Frame Wind Inputs")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Wind [m/s]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, u_wind_body, label="u Wind Body")
    axs[0, 1].plot(t, w_wind_body, label="w Wind Body")
    axs[0, 1].set_title("Body-Frame Wind Components")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Wind [m/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, altitude_error)
    axs[1, 0].set_title("Altitude Error")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("h_cmd - h [m]")
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, airspeed_error)
    axs[1, 1].set_title("Airspeed Error")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("V_cmd - V_air [m/s]")
    axs[1, 1].grid(True)

    axs[2, 0].plot(t, np.rad2deg(pitch_error), label="Pitch Error")
    axs[2, 0].plot(t, np.rad2deg(theta_error_integral), label="Integral Error")
    axs[2, 0].set_title("Pitch Error and Integral")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("deg / deg*s")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].axis("off")
    gust_type = "smooth" if config.use_smooth_gust else "square"

    scenario_text = (
        "Scenario:\n"
        f"Altitude command: {config.trim_altitude_m:.0f} m → "
        f"{config.trim_altitude_m + config.altitude_step_m:.0f} m "
        f"at t = {config.command_start_time_s:.0f} s\n"
        f"Airspeed command: {config.target_airspeed_mps:.0f} m/s\n"
        f"Gust: {config.gust_wind_z_mps:+.1f} m/s {gust_type} vertical inertial gust "
        f"from {config.gust_start_time_s:.0f}–{config.gust_end_time_s:.0f} s\n\n"
        "Key interpretation:\n"
        "The gust disturbs AoA, pitch rate, altitude, and airspeed.\n"
        "The controller recovers altitude and airspeed after the gust.\n"
        "Terminal output contains numeric performance metrics."
    )
    axs[2, 1].text(0.05, 0.55, scenario_text, fontsize=11, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(
        fig=fig,
        config=config,
        filename=f"{config.scenario_name}_validation.png",
    )


def run_simulation(
    config: ScenarioConfig,
    gains: ControllerGains = DEFAULT_GAINS,
    show_plots=True,
):
    """
    Runs the full longitudinal simulation for a given scenario config
    and controller gain set.

    State vector:
        full_state[0] = u
        full_state[1] = w
        full_state[2] = q
        full_state[3] = theta
        full_state[4] = h
        full_state[5] = theta_error_integral
        full_state[6] = delta_e_actual
        full_state[7] = delta_t_actual

    Returns:
        metrics dictionary
    """

    print_scenario_summary(config, gains)

    trim_state_5, trim_controls, trim_info = solve_trim(config.target_airspeed_mps)

    # Override altitude in trim state using scenario config.
    trim_state_5[4] = config.trim_altitude_m

    # Extend trim state:
    # [u, w, q, theta, h, theta_error_integral, delta_e_actual, delta_t_actual]
    trim_state = np.concatenate([
        trim_state_5,
        np.array([0.0]),
        np.array([trim_controls["delta_e"]]),
        np.array([trim_controls["delta_t"]]),
    ])

    print("\nTrim solution:")
    print(f"Target airspeed: {trim_info['target_airspeed']:.3f} m/s")
    print(f"Alpha trim:      {trim_info['alpha_trim_deg']:.3f} deg")
    print(f"Theta trim:      {trim_info['theta_trim_deg']:.3f} deg")
    print(f"Elevator trim:   {trim_info['delta_e_trim_deg']:.3f} deg")
    print(f"Throttle trim:   {trim_info['delta_t_trim']:.3f}")

    save_run_config(
        config=config,
        gains=gains,
        trim_info=trim_info,
        output_dir=Path(config.results_dir) / "run_configs",
        filename_prefix=config.scenario_name,
        extra_metadata={
            "simulation_type": "2D longitudinal",
            "state_vector": [
                "u",
                "w",
                "q",
                "theta",
                "h",
                "theta_error_integral",
                "delta_e_actual",
                "delta_t_actual",
            ],
            "control_architecture": {
                "longitudinal_outer_loop": "altitude error -> pitch command",
                "longitudinal_inner_loop": "pitch command -> elevator command through PID",
                "airspeed_loop": "air-relative speed error -> throttle command",
            },
            "airspeed_definition": (
                "air-relative speed computed from body velocity minus inertial wind "
                "converted into the body frame"
            ),
            "actuator_model": "first-order lag with slew-rate limiting",
            "anti_windup": (
                "pitch integral freezes when integral/elevator saturation would "
                "worsen windup"
            ),
        },
    )

    controls = make_controls(trim_controls, trim_info, config, gains)

    def dynamics_wrapper(t, full_state):
        control_values = controls(t, full_state)

        delta_e_actual = full_state[6]
        delta_t_actual = full_state[7]

        # Aircraft dynamics use the actual lagged actuator positions.
        aircraft_controls = {
            "delta_e": delta_e_actual,
            "delta_t": delta_t_actual,
            "wind_x_inertial": control_values["wind_x_inertial"],
            "wind_z_inertial": control_values["wind_z_inertial"],
        }

        aircraft_state = full_state[:5]
        aircraft_state_dot = longitudinal_dynamics(t, aircraft_state, aircraft_controls)

        theta = full_state[3]
        theta_error_integral = full_state[5]
        theta_command = control_values["theta_command"]
        theta_error = theta_command - theta

        delta_e_command = control_values["delta_e"]
        delta_t_command = control_values["delta_t"]

        # Anti-windup:
        # Freeze pitch integral if it is at its limit or if the elevator command
        # is saturated and the current error would drive it farther into saturation.
        freeze_integral = should_freeze_pitch_integral(
            theta_error=theta_error,
            theta_error_integral=theta_error_integral,
            delta_e_command=delta_e_command,
            gains=gains,
        )

        if freeze_integral:
            theta_error_integral_dot = 0.0
        else:
            theta_error_integral_dot = theta_error

        # First-order actuator lag with slew-rate limiting.
        delta_e_dot = np.clip(
            (delta_e_command - delta_e_actual) / p.ELEVATOR_TIME_CONSTANT_S,
            -p.MAX_ELEVATOR_RATE_RAD_S,
            p.MAX_ELEVATOR_RATE_RAD_S,
        )

        delta_t_dot = np.clip(
            (delta_t_command - delta_t_actual) / p.THROTTLE_TIME_CONSTANT_S,
            -p.MAX_THROTTLE_RATE_PER_S,
            p.MAX_THROTTLE_RATE_PER_S,
        )

        return np.concatenate([
            aircraft_state_dot,
            np.array([
                theta_error_integral_dot,
                delta_e_dot,
                delta_t_dot,
            ]),
        ])

    t_span = (config.sim_start_time_s, config.sim_end_time_s)
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

    t = sol.t
    u = sol.y[0]
    w = sol.y[1]
    q = sol.y[2]
    theta = sol.y[3]
    h = sol.y[4]
    theta_error_integral = sol.y[5]
    delta_e_actual = sol.y[6]
    delta_t_actual = sol.y[7]

    elevator_command = []
    throttle_command = []
    theta_command = []
    h_command = []
    v_command = []
    wind_x_inertial = []
    wind_z_inertial = []
    u_wind_body = []
    w_wind_body = []
    alpha = []
    V_air = []

    for i, time in enumerate(t):
        state_i = sol.y[:, i]
        control_i = controls(time, state_i)

        u_i = state_i[0]
        w_i = state_i[1]
        theta_i = state_i[3]

        V_air_i, alpha_i, u_wind_body_i, w_wind_body_i = compute_relative_airflow(
            u=u_i,
            w=w_i,
            theta=theta_i,
            wind_x_inertial=control_i["wind_x_inertial"],
            wind_z_inertial=control_i["wind_z_inertial"],
        )

        elevator_command.append(control_i["delta_e"])
        throttle_command.append(control_i["delta_t"])
        theta_command.append(control_i["theta_command"])
        h_command.append(control_i["h_command"])
        v_command.append(control_i["v_command"])
        wind_x_inertial.append(control_i["wind_x_inertial"])
        wind_z_inertial.append(control_i["wind_z_inertial"])
        u_wind_body.append(u_wind_body_i)
        w_wind_body.append(w_wind_body_i)
        alpha.append(alpha_i)
        V_air.append(V_air_i)

    elevator_command = np.array(elevator_command)
    throttle_command = np.array(throttle_command)
    theta_command = np.array(theta_command)
    h_command = np.array(h_command)
    v_command = np.array(v_command)
    wind_x_inertial = np.array(wind_x_inertial)
    wind_z_inertial = np.array(wind_z_inertial)
    u_wind_body = np.array(u_wind_body)
    w_wind_body = np.array(w_wind_body)
    alpha = np.array(alpha)
    V_air = np.array(V_air)

    altitude_error = h_command - h
    airspeed_error = v_command - V_air
    pitch_error = theta_command - theta

    metrics = compute_performance_metrics(
        t=t,
        h=h,
        h_command=h_command,
        V_air=V_air,
        v_command=v_command,
        theta=theta,
        theta_command=theta_command,
        elevator_command=elevator_command,
        elevator_actual=delta_e_actual,
        throttle_command=throttle_command,
        throttle_actual=delta_t_actual,
        theta_error_integral=theta_error_integral,
        config=config,
        gains=gains,
    )

    print_performance_metrics(metrics, config)

    plot_flight_response(
        t=t,
        h=h,
        h_command=h_command,
        V_air=V_air,
        v_command=v_command,
        theta=theta,
        theta_command=theta_command,
        alpha=alpha,
        q=q,
        elevator_command=elevator_command,
        elevator_actual=delta_e_actual,
        throttle_command=throttle_command,
        throttle_actual=delta_t_actual,
        config=config,
    )

    plot_validation_and_disturbance(
        t=t,
        altitude_error=altitude_error,
        airspeed_error=airspeed_error,
        pitch_error=pitch_error,
        theta_error_integral=theta_error_integral,
        wind_x_inertial=wind_x_inertial,
        wind_z_inertial=wind_z_inertial,
        u_wind_body=u_wind_body,
        w_wind_body=w_wind_body,
        config=config,
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return metrics


def main():
    """
    Entry point for a single simulation.
    """

    config = DEFAULT_SCENARIO
    gains = TUNED_GAINS

    run_simulation(
        config=config,
        gains=gains,
        show_plots=True,
    )


if __name__ == "__main__":
    main()