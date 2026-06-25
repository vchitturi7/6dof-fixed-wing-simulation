# controllers.py
# Basic controllers for longitudinal aircraft motion.

from controller_config import DEFAULT_GAINS, ControllerGains


def saturate(value, min_value, max_value):
    """
    Limits a value between min_value and max_value.
    """
    return max(min(value, max_value), min_value)


def pitch_hold_pid_controller(
    theta_command,
    theta,
    q,
    theta_error_integral,
    trim_elevator,
    gains: ControllerGains = DEFAULT_GAINS,
):
    """
    PID pitch-hold controller.

    Inputs:
        theta_command        = desired pitch angle, rad
        theta                = current pitch angle, rad
        q                    = current pitch rate, rad/s
        theta_error_integral = integral of pitch error, rad*s
        trim_elevator        = trimmed elevator deflection, rad
        gains                = controller gain settings

    Output:
        delta_e = elevator command, rad

    Sign convention:
        In this model, more negative elevator creates nose-up pitching moment.
        Therefore:
            positive pitch error -> more negative elevator
            positive accumulated error -> more negative elevator
            positive pitch rate -> less negative elevator for damping
    """

    pitch_error = theta_command - theta

    # Limit the integral term so it does not grow too much.
    theta_error_integral = saturate(
        theta_error_integral,
        -gains.max_pitch_integral_rad_s,
        gains.max_pitch_integral_rad_s,
    )

    delta_e = (
        trim_elevator
        - gains.pitch_kp * pitch_error
        - gains.pitch_ki * theta_error_integral
        + gains.pitch_kd * q
    )

    # Elevator limits.
    delta_e = saturate(
        delta_e,
        -gains.max_elevator_rad,
        gains.max_elevator_rad,
    )

    return delta_e


def altitude_hold_controller(
    h_command,
    h,
    theta_trim,
    gains: ControllerGains = DEFAULT_GAINS,
):
    """
    Simple outer-loop altitude controller.

    Inputs:
        h_command = desired altitude, m
        h         = current altitude, m
        theta_trim = trimmed pitch angle, rad
        gains     = controller gain settings

    Output:
        theta_command = desired pitch angle, rad

    Logic:
        If altitude is too low, command nose-up pitch.
        If altitude is too high, command nose-down pitch.
    """

    altitude_error = h_command - h

    theta_command = theta_trim + gains.altitude_kh_rad_per_m * altitude_error

    # Limit commanded pitch to avoid unrealistic commands.
    theta_command = saturate(
        theta_command,
        theta_trim - gains.max_pitch_command_offset_rad,
        theta_trim + gains.max_pitch_command_offset_rad,
    )

    return theta_command


def airspeed_hold_controller(
    v_command,
    v,
    trim_throttle,
    gains: ControllerGains = DEFAULT_GAINS,
):
    """
    Simple P airspeed controller.

    Inputs:
        v_command     = desired airspeed, m/s
        v             = current airspeed, m/s
        trim_throttle = trimmed throttle command, 0 to 1
        gains         = controller gain settings

    Output:
        delta_t = throttle command, 0 to 1

    Logic:
        If airspeed is too low, increase throttle.
        If airspeed is too high, reduce throttle.
    """

    airspeed_error = v_command - v

    delta_t = trim_throttle + gains.airspeed_kv * airspeed_error

    delta_t = saturate(
        delta_t,
        gains.min_throttle,
        gains.max_throttle,
    )

    return delta_t