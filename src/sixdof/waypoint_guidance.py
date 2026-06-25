# waypoint_guidance.py
# Line-segment path-following guidance for the 6DOF fixed-wing simulation.
#
# Purpose:
#   Converts aircraft position into:
#       heading command
#       altitude command
#       airspeed command
#
# Architecture:
#
#   Waypoints
#       -> active path leg
#       -> path heading + cross-track correction
#       -> heading / altitude / airspeed commands
#       -> 6DOF autopilot
#
# This improves on simple point-chasing guidance.
#
# Point chasing:
#   aircraft points directly at the next waypoint.
#   This can cause orbiting around the final waypoint.
#
# Path following:
#   aircraft tracks the line segment between waypoints.
#   This produces smoother, more realistic fixed-wing navigation.
#
# Coordinate convention:
#   Inertial frame:
#       +x = North / forward
#       +y = East / right
#       +z = Down
#
# Altitude:
#       altitude = -z_down

from dataclasses import dataclass
import numpy as np

from rotations import wrap_angle_pi


@dataclass
class Waypoint:
    """
    One navigation waypoint.
    """

    x_m: float
    y_m: float
    altitude_m: float
    airspeed_mps: float = 25.0


@dataclass
class WaypointGuidanceConfig:
    """
    Configuration for line-segment path-following guidance.
    """

    waypoint_capture_radius_m: float = 40.0
    cross_track_gain_rad_per_m: float = 0.009
    max_cross_track_correction_rad: float = np.deg2rad(45.0)
    max_heading_change_per_command_rad: float = np.deg2rad(55.0)
    leg_overshoot_margin_m: float = 10.0
    final_waypoint_capture_radius_m: float = 75.0


def get_position_xy(state):
    """
    Returns aircraft horizontal position [x, y].
    """

    return np.array([
        state[0],
        state[1],
    ])


def waypoint_xy(waypoint: Waypoint):
    """
    Returns waypoint horizontal position [x, y].
    """

    return np.array([
        waypoint.x_m,
        waypoint.y_m,
    ])


def horizontal_distance_to_waypoint(state, waypoint: Waypoint):
    """
    Computes horizontal distance from aircraft to waypoint.
    """

    aircraft_xy = get_position_xy(state)
    target_xy = waypoint_xy(waypoint)

    return np.linalg.norm(target_xy - aircraft_xy)


def bearing_to_waypoint(state, waypoint: Waypoint):
    """
    Computes direct bearing from aircraft to waypoint.

    This is still useful for diagnostics and fallback behavior.
    """

    aircraft_xy = get_position_xy(state)
    target_xy = waypoint_xy(waypoint)

    delta = target_xy - aircraft_xy

    return np.arctan2(delta[1], delta[0])


def get_active_leg_indices(current_waypoint_index, number_of_waypoints):
    """
    Returns the start and end waypoint indices for the active path leg.

    If current_waypoint_index = 0:
        active leg is WP0 -> WP1 after WP0 is reached.
        But before WP0, we use aircraft -> WP0 behavior by setting both
        previous/current logic carefully in compute_path_following_commands.

    For index >= 1:
        active leg is WP(index-1) -> WP(index).
    """

    if number_of_waypoints < 2:
        raise ValueError("Path following requires at least two waypoints.")

    end_index = min(
        current_waypoint_index,
        number_of_waypoints - 1,
    )

    start_index = max(
        end_index - 1,
        0,
    )

    return start_index, end_index


def compute_line_segment_geometry(state, start_waypoint, end_waypoint):
    """
    Computes line-segment path geometry.

    Inputs:
        state          = aircraft 12-state vector
        start_waypoint = beginning of active path leg
        end_waypoint   = end of active path leg

    Returns:
        dictionary containing:
            path_heading_rad
            cross_track_error_m
            along_track_distance_m
            leg_length_m
            progress_fraction
            distance_to_end_m

    Cross-track sign convention:
        Positive cross-track error means aircraft is left of the path
        with respect to the direction from start waypoint to end waypoint.

        The heading correction uses:
            heading_command = path_heading - correction

        So positive cross-track error commands a right correction.
    """

    aircraft_xy = get_position_xy(state)

    start_xy = waypoint_xy(start_waypoint)
    end_xy = waypoint_xy(end_waypoint)

    leg_vector = end_xy - start_xy
    leg_length_m = np.linalg.norm(leg_vector)

    if leg_length_m < 1e-6:
        raise ValueError("Waypoint leg length is too small.")

    path_unit = leg_vector / leg_length_m

    path_heading_rad = np.arctan2(
        path_unit[1],
        path_unit[0],
    )

    relative_position = aircraft_xy - start_xy

    along_track_distance_m = np.dot(
        relative_position,
        path_unit,
    )

    # 2D cross product: path_unit x relative_position.
    # Positive means aircraft is left of path.
    cross_track_error_m = (
        path_unit[0] * relative_position[1]
        - path_unit[1] * relative_position[0]
    )

    progress_fraction = along_track_distance_m / leg_length_m

    distance_to_end_m = np.linalg.norm(
        end_xy - aircraft_xy
    )

    return {
        "path_heading_rad": path_heading_rad,
        "cross_track_error_m": cross_track_error_m,
        "along_track_distance_m": along_track_distance_m,
        "leg_length_m": leg_length_m,
        "progress_fraction": progress_fraction,
        "distance_to_end_m": distance_to_end_m,
    }


def should_advance_waypoint(
    state,
    waypoints,
    current_waypoint_index,
    config: WaypointGuidanceConfig,
):
    """
    Decides whether to advance to the next waypoint.

    Switches when:
        1. aircraft is inside capture radius of active waypoint, or
        2. aircraft has passed beyond the end of the active leg.

    The final waypoint is not advanced beyond the final index.
    """

    number_of_waypoints = len(waypoints)

    if current_waypoint_index >= number_of_waypoints - 1:
        return False

    active_waypoint = waypoints[current_waypoint_index]

    distance_to_active = horizontal_distance_to_waypoint(
        state=state,
        waypoint=active_waypoint,
    )

    if distance_to_active <= config.waypoint_capture_radius_m:
        return True

    if current_waypoint_index >= 1:
        start_index, end_index = get_active_leg_indices(
            current_waypoint_index=current_waypoint_index,
            number_of_waypoints=number_of_waypoints,
        )

        geometry = compute_line_segment_geometry(
            state=state,
            start_waypoint=waypoints[start_index],
            end_waypoint=waypoints[end_index],
        )

        if (
            geometry["along_track_distance_m"]
            >= geometry["leg_length_m"] + config.leg_overshoot_margin_m
        ):
            return True

    return False


def update_waypoint_index(
    state,
    waypoints,
    current_waypoint_index,
    config: WaypointGuidanceConfig,
):
    """
    Updates active waypoint index.

    This function advances at most one waypoint per call.
    """

    if should_advance_waypoint(
        state=state,
        waypoints=waypoints,
        current_waypoint_index=current_waypoint_index,
        config=config,
    ):
        return min(
            current_waypoint_index + 1,
            len(waypoints) - 1,
        )

    return current_waypoint_index


def limit_heading_command_change(
    current_heading_rad,
    desired_heading_rad,
    max_heading_change_rad,
):
    """
    Limits heading command relative to current aircraft heading.
    """

    heading_error = wrap_angle_pi(
        desired_heading_rad - current_heading_rad
    )

    heading_error_limited = max(
        min(heading_error, max_heading_change_rad),
        -max_heading_change_rad,
    )

    return wrap_angle_pi(
        current_heading_rad + heading_error_limited
    )


def compute_path_following_heading(
    state,
    waypoints,
    current_waypoint_index,
    config: WaypointGuidanceConfig,
):
    """
    Computes heading command using line-segment path following.

    For current_waypoint_index == 0:
        The aircraft has not yet reached WP0, so it points directly to WP0.

    For current_waypoint_index >= 1:
        Track the line segment:
            WP(current_index - 1) -> WP(current_index)
    """

    current_heading_rad = state[8]

    number_of_waypoints = len(waypoints)

    if current_waypoint_index == 0:
        active_waypoint = waypoints[0]

        desired_heading_rad = bearing_to_waypoint(
            state=state,
            waypoint=active_waypoint,
        )

        heading_command_rad = limit_heading_command_change(
            current_heading_rad=current_heading_rad,
            desired_heading_rad=desired_heading_rad,
            max_heading_change_rad=config.max_heading_change_per_command_rad,
        )

        distance_to_waypoint_m = horizontal_distance_to_waypoint(
            state=state,
            waypoint=active_waypoint,
        )

        return {
            "heading_command_rad": heading_command_rad,
            "desired_heading_rad": desired_heading_rad,
            "path_heading_rad": desired_heading_rad,
            "cross_track_error_m": 0.0,
            "along_track_distance_m": 0.0,
            "leg_length_m": distance_to_waypoint_m,
            "progress_fraction": 0.0,
            "distance_to_waypoint_m": distance_to_waypoint_m,
        }

    start_index, end_index = get_active_leg_indices(
        current_waypoint_index=current_waypoint_index,
        number_of_waypoints=number_of_waypoints,
    )

    geometry = compute_line_segment_geometry(
        state=state,
        start_waypoint=waypoints[start_index],
        end_waypoint=waypoints[end_index],
    )

    path_heading_rad = geometry["path_heading_rad"]
    cross_track_error_m = geometry["cross_track_error_m"]

    cross_track_correction_rad = (
        config.cross_track_gain_rad_per_m
        * cross_track_error_m
    )

    cross_track_correction_rad = max(
        min(
            cross_track_correction_rad,
            config.max_cross_track_correction_rad,
        ),
        -config.max_cross_track_correction_rad,
    )

    desired_heading_rad = wrap_angle_pi(
        path_heading_rad - cross_track_correction_rad
    )

    # Final waypoint behavior:
    # Once close to the final waypoint, stop trying to point directly back at it.
    # Continue roughly along the final path heading. This prevents terminal orbiting.
    is_final_waypoint = current_waypoint_index >= number_of_waypoints - 1

    if (
        is_final_waypoint
        and geometry["distance_to_end_m"] <= config.final_waypoint_capture_radius_m
    ):
        desired_heading_rad = path_heading_rad

    heading_command_rad = limit_heading_command_change(
        current_heading_rad=current_heading_rad,
        desired_heading_rad=desired_heading_rad,
        max_heading_change_rad=config.max_heading_change_per_command_rad,
    )

    return {
        "heading_command_rad": heading_command_rad,
        "desired_heading_rad": desired_heading_rad,
        "path_heading_rad": path_heading_rad,
        "cross_track_error_m": cross_track_error_m,
        "along_track_distance_m": geometry["along_track_distance_m"],
        "leg_length_m": geometry["leg_length_m"],
        "progress_fraction": geometry["progress_fraction"],
        "distance_to_waypoint_m": geometry["distance_to_end_m"],
    }


def compute_waypoint_commands(
    state,
    waypoints,
    current_waypoint_index,
    config: WaypointGuidanceConfig,
):
    """
    Computes autopilot commands for path-following guidance.

    Returns:
        commands dictionary:
            altitude_command_m
            airspeed_command_mps
            heading_command_rad

        guidance_info dictionary:
            active_waypoint_index
            active_waypoint_x_m
            active_waypoint_y_m
            distance_to_waypoint_m
            desired_heading_rad
            path_heading_rad
            heading_command_rad
            cross_track_error_m
            progress_fraction
    """

    if not waypoints:
        raise ValueError("Waypoint list cannot be empty.")

    if len(waypoints) < 2:
        raise ValueError("Path following requires at least two waypoints.")

    current_waypoint_index = update_waypoint_index(
        state=state,
        waypoints=waypoints,
        current_waypoint_index=current_waypoint_index,
        config=config,
    )

    active_waypoint = waypoints[current_waypoint_index]

    heading_info = compute_path_following_heading(
        state=state,
        waypoints=waypoints,
        current_waypoint_index=current_waypoint_index,
        config=config,
    )

    commands = {
        "altitude_command_m": active_waypoint.altitude_m,
        "airspeed_command_mps": active_waypoint.airspeed_mps,
        "heading_command_rad": heading_info["heading_command_rad"],
    }

    guidance_info = {
        "active_waypoint_index": current_waypoint_index,
        "active_waypoint_x_m": active_waypoint.x_m,
        "active_waypoint_y_m": active_waypoint.y_m,
        "active_waypoint_altitude_m": active_waypoint.altitude_m,
        "distance_to_waypoint_m": heading_info["distance_to_waypoint_m"],
        "desired_heading_rad": heading_info["desired_heading_rad"],
        "path_heading_rad": heading_info["path_heading_rad"],
        "heading_command_rad": heading_info["heading_command_rad"],
        "cross_track_error_m": heading_info["cross_track_error_m"],
        "along_track_distance_m": heading_info["along_track_distance_m"],
        "leg_length_m": heading_info["leg_length_m"],
        "progress_fraction": heading_info["progress_fraction"],
    }

    return commands, guidance_info


def make_default_waypoints():
    """
    Creates a simple waypoint course.

    These waypoints are intentionally reachable with smooth fixed-wing turns.
    """

    return [
        Waypoint(
            x_m=500.0,
            y_m=0.0,
            altitude_m=100.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=900.0,
            y_m=250.0,
            altitude_m=120.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1300.0,
            y_m=-150.0,
            altitude_m=120.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1700.0,
            y_m=0.0,
            altitude_m=140.0,
            airspeed_mps=26.0,
        ),
    ]


if __name__ == "__main__":
    test_state = np.array([
        0.0,     # x
        0.0,     # y
        -100.0,  # z_down
        25.0,    # u
        0.0,     # v
        0.0,     # w
        0.0,     # phi
        0.0,     # theta
        0.0,     # psi
        0.0,     # p
        0.0,     # q
        0.0,     # r
    ])

    waypoints = make_default_waypoints()
    config = WaypointGuidanceConfig()

    commands, guidance_info = compute_waypoint_commands(
        state=test_state,
        waypoints=waypoints,
        current_waypoint_index=0,
        config=config,
    )

    print("Path-following guidance test")
    print("----------------------------")
    print("Commands:")
    print(commands)
    print("\nGuidance info:")
    print(guidance_info)