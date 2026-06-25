# waypoint_course_config.py
# Easy-to-edit waypoint course definitions for the 6DOF waypoint-following sim.
#
# Edit this file when you want to test different waypoint layouts.
#
# Coordinate convention:
#   x_m = north / forward position [m]
#   y_m = east / right position [m]
#   altitude_m = altitude above ground/reference [m]
#   airspeed_mps = commanded airspeed at that waypoint [m/s]

from waypoint_guidance import Waypoint


def make_default_waypoints():
    """
    Final baseline course.

    This is the current polished waypoint-following demo:
        WP0 -> WP1 -> WP2 -> WP3
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


def make_easy_course():
    """
    Easier course with gentler turns.

    Use this when you want a cleaner-looking path with less corner cutting.
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
            y_m=150.0,
            altitude_m=115.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1300.0,
            y_m=100.0,
            altitude_m=125.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1700.0,
            y_m=0.0,
            altitude_m=135.0,
            airspeed_mps=25.5,
        ),
    ]


def make_aggressive_course():
    """
    Harder course with sharper heading changes.

    Use this to show the aircraft has realistic turn-radius limitations.
    """

    return [
        Waypoint(
            x_m=500.0,
            y_m=0.0,
            altitude_m=100.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=850.0,
            y_m=350.0,
            altitude_m=120.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1250.0,
            y_m=-300.0,
            altitude_m=120.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1750.0,
            y_m=100.0,
            altitude_m=145.0,
            airspeed_mps=26.0,
        ),
    ]


def make_straight_climb_course():
    """
    Simple course for testing altitude and airspeed behavior without big turns.
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
            y_m=0.0,
            altitude_m=115.0,
            airspeed_mps=25.0,
        ),
        Waypoint(
            x_m=1300.0,
            y_m=0.0,
            altitude_m=130.0,
            airspeed_mps=25.5,
        ),
        Waypoint(
            x_m=1700.0,
            y_m=0.0,
            altitude_m=140.0,
            airspeed_mps=26.0,
        ),
    ]


def get_waypoint_course(course_name):
    """
    Selects a waypoint course by name.

    Valid course names:
        "default"
        "easy"
        "aggressive"
        "straight_climb"
    """

    course_name = course_name.lower()

    if course_name == "default":
        return make_default_waypoints()

    if course_name == "easy":
        return make_easy_course()

    if course_name == "aggressive":
        return make_aggressive_course()

    if course_name == "straight_climb":
        return make_straight_climb_course()

    raise ValueError(
        f"Unknown waypoint course: {course_name}. "
        "Choose from: default, easy, aggressive, straight_climb."
    )