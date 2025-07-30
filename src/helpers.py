import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import json
import numpy as np
import websockets
from dataclasses import dataclass


###########--------------------------- JSON Utilities ---------------------------###########

def validate_json_msg(json_text: str) -> None:
    """Verify that the input message is a valid JSON string.
    Args: 
        json_text (str): The input string to validate as JSON.
    Returns: 
        None
    Raises: 
        ValueError: If the input string is not valid JSON.
    """
    try:
        json.loads(json_text)
        return
    except json.JSONDecodeError as e:
        raise ValueError(f"Did not receive a valid json velocity input to send to the server: {json_text}")


def to_json(input_list: list[float] = []) -> str:
    """Converts a list of two float velocities to a JSON message for the server.
    Args:
        input_list (list[float]): List of two float values [v_left, v_right].
    Returns:
        str: JSON-formatted string with keys 'v_left' and 'v_right'.
    Raises:
        ValueError: If input_list does not have exactly two elements.
    """
    if len(input_list) != 2:
        raise ValueError(f"Input list must be of length 2. Received {input_list}")
    
    v_left, v_right = input_list
    msg = {"v_left": v_left, "v_right": v_right}
    json_msg = json.dumps(msg)
    validate_json_msg(json_msg)
    
    return json_msg


def from_json(json_msg: str) -> list[float]:
    """Converts a JSON message with velocity inputs to a list of floats.
    Args:
        json_msg (str): JSON-formatted string containing 'v_left' and 'v_right'.
    Returns:
        list[float]: List containing [v_left, v_right] as floats.
    Raises:
        ValueError: If required keys are missing or input is not valid JSON.
    """
    validate_json_msg(json_msg)
    msg = json.loads(json_msg)
    if "v_left" not in msg or "v_right" not in msg:
        raise ValueError(f"Input json message must contain v_left and v_right keys. Received {json_msg}")
    return [msg["v_left"], msg["v_right"]]


###########--------------------------- WebSocket Utilities ---------------------------###########


async def connect() -> websockets.WebSocketClientProtocol:
    """Connects to the WebSocket server and returns the connection object.
    Args:
        uri (str) : The uri of the websocket
    Returns:
        websockets.WebSocketClientProtocol: The connected websocket client protocol object.
    """
    
    uri = "ws://91.99.103.188:8765"
    websocket = await websockets.connect(uri)
    print("Connected to the websocket!")
    return websocket


###########--------------------------- Plotting Utilities ---------------------------###########


@dataclass
class PlotHandles:
    fig: Figure
    ax: Axes
    gps_current_pose: Line2D
    robot_current_pose: Line2D
    gps_line: Line2D
    robot_line: Line2D
    ref_line: Line2D

def initialize_live_plot(ref_traj_name: str) -> PlotHandles:
    """Initializes a live plot for robot tracking, including reference trajectory, GPS data, and EKF estimates.
    
    Args:
        ref_traj_name (str): Name of the reference trajectory. This is used to set the axes limits.
    Returns:
        PlotHandles: Object containing figure, axis, and Line2D handles for live plotting.
    """
    
    plt.ion()
    
    fig, ax = plt.subplots()

    # Main plot: Where the robot should be going vs where the robot is going?
    # Plots the reference trajectory in green, GPS measurements path in blue and
    # the robot's estimated path in magenta. To help visualize the current state of the 
    # robot, this plots a yellow dot at the current robot (x, y) pose and plots a 
    # red dot for the current GPS (x, y) pose
    sc, = ax.plot([], [], 'ro', label='GPS Current Pose')  # red dot for GPS
    es, = ax.plot([], [], 'yo', label='EKF Current Pose')  # yellow dot for EKF
    gps_line, = ax.plot([], [], 'b-', label='GPS Path')  # blue line for GPS path
    robot_line, = ax.plot([], [], 'm-', label='Robot Path')  # magenta line for robot path
    ref_line, = ax.plot([], [], 'g-', label='Reference')  # green line for reference
    ax.set_xlabel("X in (m)")
    ax.set_ylabel("Y in (m)")
    ax.grid(True)
    ax.set_title("Real-Time XY Coordinates")
    ax.legend()
    if (ref_traj_name == "course-8") or (ref_traj_name == "semi-circle"):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 8)
    else:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)


    return PlotHandles(
        fig=fig,
        ax=ax,
        gps_current_pose=sc,
        robot_current_pose=es,
        gps_line=gps_line,
        robot_line=robot_line,
        ref_line=ref_line,
    )

def plot_live_data(
    handles: PlotHandles,
    gps_data: list[list[float]] = [],
    estimated_poses: list[list[float]] = [],
    reference_poses: list[list[float]] = [],
) -> None:
    """Plots live data for GPS, EKF estimated poses, and reference trajectory on a pre-initialized plot.
    Args:
        handles (PlotHandles)              : Plot handles containing figure, axis, and Line2D objects.
        gps_data (list[list[float]])       : List of [x, y] GPS positions.
        estimated_poses (list[list[float]]): List of estimated robot states [x, y, θ, v, ω] per entry.
        reference_poses (list[list[float]]): List of reference trajectory poses, at least [x, y] per entry.
    Returns:
        None
    """

    # Unpack member variables of the PlotHandles object
    fig = handles.fig
    sc = handles.gps_current_pose
    es = handles.robot_current_pose
    gps_line = handles.gps_line
    robot_line = handles.robot_line
    ref_line = handles.ref_line

    gps_data_live = gps_data[:]
    estimated_poses_live = estimated_poses[:]
    reference_poses_live = reference_poses[:]

    # Update the live plot
    if gps_data_live:
        gps_arr = np.array(gps_data_live)
        # Draw GPS path and the current GPS point
        gps_line.set_data(gps_arr[:, 0], gps_arr[:, 1])
        sc.set_data([gps_arr[-1, 0]], [gps_arr[-1, 1]])
    if estimated_poses_live:
        est_arr = np.array(estimated_poses_live)
        es.set_data([est_arr[-1, 0]], [est_arr[-1, 1]])
        # Draw robot path and the current robot point
        if est_arr.shape[1] >= 2:
            robot_line.set_data(est_arr[:, 0], est_arr[:, 1])
    # Update reference trajectory
    if reference_poses_live:
        # reference_poses is now a list of lists; plot first two elements of each
        ref_arr = np.array([p[:2] for p in reference_poses_live])
        if ref_arr.shape[1] == 2:
            ref_line.set_data(ref_arr[:, 0], ref_arr[:, 1])

    # Redraw
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_x_y_thetha(
    reference_poses: list[list[float]],
    reference_heading: list[float],
    estimated_poses: list[list[float]],
    time_stamps: list[float],
    ref_traj_name: str
) -> None:
    """Plots robot and reference trajectory heading, X, and Y positions over time.
    Args:
        reference_poses (list[list[float]]) : List of reference trajectory poses, at least [x, y] per entry.
        reference_heading (list[float])     : List of reference heading angles in (radians).
        estimated_poses (list[list[float]]) : List of estimated robot states [x, y, θ, v, ω] per entry.
        time_stamps (list[float])           : List of time stamps corresponding to the data.
        ref_traj_name (str)                 : Name of the reference trajectory. This is used to set the axes limits.
    Returns:
        None
    """

    fig, axs = plt.subplots(1, 3, figsize=(8, 12))
    ax_heading = axs[0]
    ax_x = axs[1]
    ax_y = axs[2]

    # Subplot (0): Robot Heading vs Reference Heading
    reference_heading_line, = ax_heading.plot([], [], 'g-', label='Reference trajectory heading')
    robot_heading_line, = ax_heading.plot([], [], 'm-', label='Robot heading')
    ax_heading.set_title("Robot Heading vs Reference Heading")
    ax_heading.set_xlim(0, 20)
    ax_heading.set_ylim(-180, 180)
    ax_heading.set_xlabel("Time in (sec)")
    ax_heading.set_ylabel("Theta in (deg)")
    ax_heading.legend()

    # Subplot (1): Reference trajectory X position vs robot's X position
    ref_x_line, = ax_x.plot([], [], 'g-', label='Reference trajectory X')
    robot_x_line, = ax_x.plot([], [], 'm-', label='Robot X')
    ax_x.set_title("Reference X vs Robot X")
    ax_x.set_xlim(0, 20)
    if ref_traj_name == "course-8":
        ax_x.set_ylim(-3, 3)
    elif ref_traj_name == "semi-circle":
        ax_x.set_ylim(-5, 5)
    else:
        ax_x.set_ylim(0, 8)
    ax_x.set_xlabel("Time (sec)")
    ax_x.set_ylabel("X in (m)")
    ax_x.legend()

    # Subplot (2): Reference trajectory Y position vs robot's Y position
    ref_y_line, = ax_y.plot([], [], 'g-', label='Reference Y')
    robot_y_line, = ax_y.plot([], [], 'm-', label='Robot Y')
    ax_y.set_title("Reference Y vs Robot Y")
    ax_y.set_xlim(0, 20)
    if ref_traj_name == "course-8":
        ax_y.set_ylim(-3, 3)
    elif ref_traj_name == "semi-circle":
        ax_y.set_ylim(-5, 5)
    else:
        ax_y.set_ylim(0, 8)
    ax_y.set_xlabel("Time (sec)")
    ax_y.set_ylabel("Y in (m)")
    ax_y.legend()

    estimated_poses_live = estimated_poses[:]
    reference_poses_live = reference_poses[:]
    reference_heading_live = reference_heading[:]

    # Plot robot estimated data
    if estimated_poses_live:
        est_arr = np.array(estimated_poses_live)
        if est_arr.shape[1] >= 2:
            t_arr = np.array(time_stamps)
            robot_x_line.set_data(t_arr[:], est_arr[:, 0])
            robot_y_line.set_data(t_arr[:], est_arr[:, 1])
            robot_heading_line.set_data(t_arr[:], np.rad2deg(est_arr[:, 2]))
    # Plot reference trajectory data
    if reference_poses_live:
        ref_arr = np.array([p[:2] for p in reference_poses_live])
        t_arr = np.array(time_stamps)
        if ref_arr.shape[1] == 2:
            ref_x_line.set_data(t_arr[:], ref_arr[:, 0])
            ref_y_line.set_data(t_arr[:], ref_arr[:, 1])
    if reference_heading_live:
        t_arr = np.array(time_stamps)
        reference_heading_arr = np.array(reference_heading_live)
        reference_heading_line.set_data(t_arr[:], np.rad2deg(reference_heading_arr[:]))
    plt.ioff()  # Turn off interactive mode to ensure plt.show() blocks
    plt.show(block=True)  # Block until the plot window is closed
