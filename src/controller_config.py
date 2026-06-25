# controller_config.py
# Stores controller gain settings for the longitudinal autopilot.
#
# Keeping gains in one place makes it much easier to tune the controller
# and compare different control designs.

from dataclasses import dataclass


@dataclass
class ControllerGains:
    """
    Controller gains for the longitudinal autopilot.

    Pitch controller:
        delta_e = trim_elevator - Kp_theta * theta_error
                            - Ki_theta * integral(theta_error)
                            + Kd_q * q

    Altitude controller:
        theta_command = theta_trim + Kh * altitude_error

    Airspeed controller:
        throttle = trim_throttle + Kv * airspeed_error
    """

    # Pitch PID gains
    pitch_kp: float = 1.0
    pitch_ki: float = 0.12
    pitch_kd: float = 0.35

    # Altitude outer-loop gain
    altitude_kh_rad_per_m: float = 0.004363323129985824
    # This equals np.deg2rad(0.25)

    # Airspeed/throttle gain
    airspeed_kv: float = 0.08

    # Limits
    max_pitch_integral_rad_s: float = 0.5235987755982988
    # This equals np.deg2rad(30.0)

    max_elevator_rad: float = 0.4363323129985824
    # This equals np.deg2rad(25.0)

    min_throttle: float = 0.0
    max_throttle: float = 1.0

    max_pitch_command_offset_rad: float = 0.17453292519943295
    # This equals np.deg2rad(10.0)


DEFAULT_GAINS = ControllerGains()

# Tuned gains selected from gain sweep (run_gain_sweep.py).
# Winner: stronger_airspeed_hold — best overall score (46.40 vs 48.11 default).
# Change from default: airspeed_kv 0.08 → 0.15.
# Effect: airspeed error reduced from 1.29 m/s to 0.93 m/s,
#         airspeed settling time reduced from 26.7 s to 20.4 s.
# Tradeoff: altitude settling slightly slower (35.5 s vs 29.6 s).
TUNED_GAINS = ControllerGains(
    airspeed_kv=0.15,
)