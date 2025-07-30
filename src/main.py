"""Main file that establishes connection with the webscoket, runs the robot simulations, displays 
relevant plots for robot trajectory tracking and closes the websocket connection."""

import asyncio
import websockets
import numpy as np
import json
import time
from typing import List
from dataclasses import dataclass
from controller import DynamicCompensator
from reference_traj import ( 
    get_ref_figure_eight, get_ref_simple_curve, get_ref_straight_line)
from helpers import (
    connect, initialize_live_plot, to_json, plot_live_data, plot_x_y_thetha, PlotHandles
)
from ekf import ExtendedKalmanFilter


########## ------------------------- Const variables and helper classes ------------------------##########

# ---- Robot and sim specifics ----
# Distance between the wheels of the robot
ROBOT_WIDTH = 0.5 # m
# Because we get out score after 20 seconds from the start of the simulation.
# close the connection to the webscoket after 20 seconds
END_SIM = 20 # seconds
# Physical constraints for the robot
ACCELERATION_LIMITS = [-1.0, 1.0]
VELOCITY_LIMITS = [-2.0, 2.0]
# You can try the controller + ekf setup with different reference trajectories! 
# Possible options are "straight-line", "semi-circle"
REF_TRAJ_NAME = "course-8"

# ---- Initial values ----
# We could initialize this based on the reference trajectory we have, but for simplicity initialize it to 0
REF_TRAJ_INITIAL = np.zeros(6) # [x_ref, y_ref, xd_ref, yd_ref, xdd_ref, ydd_ref]
U_INITIAL = np.array([0.0, 0.0]) # [v_left, v_right]
# We know that the robot starts at the origin, pointing towards X-axis, i.e theta = 0
X_INITIAL = np.zeros(5) # [x, y, θ, v, ω]
# These are only initial process covariances for our 5x1 states. EKF updates the covariances through the simulation
P_INITIAL = np.diag([1.0**2, 1.0**2, np.deg2rad(10.0)**2, 0.5**2, 0.2**2])

# ---- Controller specifics ----
# The values have been tuned to prioritize tracking and reduce oscillations 
K_P = 1.35
K_D = 5.7


@dataclass
class RobotState:
    """Groups all robot state variables"""
    X: np.array  # Current estimated state [x, y, theta, v, omega] (5x1)
    P: np.array  # Current process covariance matrix (5x5)
    U: np.array  # Current control command [v_left, v_right] (2x1)
    ref_traj: np.array  # Current reference trajectory [x_ref, y_ref, xd_ref, yd_ref, xdd_ref, ydd_ref] (6x1)

@dataclass
class TimeState:
    """Groups all time-related variables"""
    t_current: float
    t_start: float
    t_prev: float

@dataclass
class DataBuffers:
    """Groups all data lists. Contains sensor data, reference trajectory data and estimated pose data"""
    # Sensor data
    accel_data: List
    gyro_data: List
    gps_data: List
    # Robot estimated poses data
    est_poses: List
    # Reference trajectory data
    ref_poses: List
    ref_heading: List
    # Time history
    time_stamps: List

########## ------------------------- Update co-routine ------------------------------##########


async def update(websocket, 
                 time_state: TimeState, 
                 robot_state: RobotState, 
                 controller: DynamicCompensator, 
                 ekf: ExtendedKalmanFilter, 
                 buffers: DataBuffers, 
                 plot_handles: PlotHandles) -> tuple[TimeState, RobotState]:
    """
    Performs one iteration of the robot control and state estimation loop to track the reference trajectory.

    Args:
        websocket: The WebSocket connection object.
        time_state (TimeState): Object containing time-related variables.
        robot_state (RobotState): Object containing robot state variables.
        controllers (DynamicCompensator): Object containing an initialized dynamic-compensator.
        ekf (ExtendedKalmanFilter): Object containing an initialized extended kalman filter
        buffers (SensorBuffers): Object containing all sensor data buffers.
        plot_handles: Handles for the live plot.

    Returns:
        tuple: A tuple containing the updated state variables:
               (time_state: TimeState, robot_state: RobotState)
    """
    try:
        msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
        data = json.loads(msg)
    except asyncio.TimeoutError:
        print("Skipping state. Didn't receive serve data")
        return time_state, robot_state  # Skip this frame gracefully

    if data.get("message_type") == "sensors":
        # If the message received is of "sensors" type log all the information
        # This is useful for plotting sensor observations later
        time_state.t_current = time.time()
        buffers.est_poses.append(robot_state.X.copy())
        buffers.ref_poses.append(robot_state.ref_traj)
        buffers.ref_heading.append(np.arctan2(robot_state.ref_traj[3], robot_state.ref_traj[2]))
        buffers.time_stamps.append(time_state.t_current - time_state.t_start)

        for sensor in data["sensors"]:
            name = sensor["name"]
            sensor_values = np.array(sensor["data"])

            if name == "accelerometer":
                buffers.accel_data.append(sensor_values)
            elif name == "gyro":
                buffers.gyro_data.append(sensor_values)
            elif name == "gps":
                buffers.gps_data.append(sensor_values)

        is_gps_updated = True
        if len(buffers.gps_data) > 2:
            if np.array_equal(buffers.gps_data[-2], buffers.gps_data[-1]):
                is_gps_updated = False

        # Gather all the latest sensor measurements into a list
        z = [buffers.gps_data[-1][0], buffers.gps_data[-1][1], buffers.accel_data[-1][0], 
             buffers.accel_data[-1][1], buffers.gyro_data[-1][0]]
        
        # Compute dt
        now = time.time()
        dt = now - time_state.t_prev
        time_state.t_prev = now

        # Run EKF update on the robot
        robot_state.X, robot_state.P = ekf.update(robot_state.X, robot_state.U, 
                                                             robot_state.P, dt, z, is_gps_updated)
        # Update the reference trajectory
        if REF_TRAJ_NAME == "course-8":
            robot_state.ref_traj = get_ref_figure_eight(time_state.t_current - time_state.t_start)
        elif REF_TRAJ_NAME == "straight-line":
            robot_state.ref_traj = get_ref_straight_line(time_state.t_current - time_state.t_start)
        elif REF_TRAJ_NAME == "semi-circle":
            robot_state.ref_traj = get_ref_simple_curve(time_state.t_current - time_state.t_start)
        else:
            raise ValueError(f"Did not receive a valid reference trajectory name: {REF_TRAJ_NAME}")
        
        # Compute the control commands using the controller
        v_left, v_right = controller.update(robot_state.X, robot_state.ref_traj, dt)
        robot_state.U = np.array([v_left, v_right])
        # Send the commands to the server
        await websocket.send(to_json(robot_state.U.tolist()))           

    # Check if we are receiving any other messages from the server
    # This could be the L2 norm!
    elif data.get("message_type") == "score":
        print("\nReceived L2 score :", data["score"], "\n")
    
    # Visualize the live-plot as the robot moves
    plot_live_data(handles=plot_handles,
                   gps_data=buffers.gps_data,
                   estimated_poses=buffers.est_poses,
                   reference_poses=buffers.ref_poses)

    # Return the updated mutable states
    return time_state, robot_state


########## --------------------------------- Main ------------------------------------##########


async def main():
    # Connect to the websocket
    websocket = await connect()
    try:
        # Initialize live-plot. This needs to happen only once
        plot_handles = initialize_live_plot()

        # Timing setup
        t_start = time.time()
        # This is initialized here and updated in the update function call
        t_current = t_start 

        # Sensor, reference, time and robot state data
        buffers = DataBuffers(
            accel_data=[],
            gyro_data=[],
            gps_data=[],
            est_poses=[],
            ref_poses=[],
            ref_heading=[],
            time_stamps=[]
        )

        # Initialize the reference trajectory for t=0
        initial_ref_traj = REF_TRAJ_INITIAL # Get initial reference values
        
        X = X_INITIAL  # [x, y, theta, v, omega]
        # Initialize process covariance
        P = P_INITIAL
        # Send initial control commands to start the simulation
        U = U_INITIAL
        await websocket.send(to_json(U.tolist()))

        robot_state = RobotState(X=X, P=P, U=U, ref_traj=initial_ref_traj)

        # Initialize the controller
        controller = DynamicCompensator(kp=K_P, 
                                        kd=K_D, 
                                        robot_width=ROBOT_WIDTH,
                                        acceleration_limit=ACCELERATION_LIMITS,
                                        velocity_limits=VELOCITY_LIMITS,
                                        initial_u1=U.mean()) 
        
        # Initialize the extended kalman filter
        ekf = ExtendedKalmanFilter(robot_width=ROBOT_WIDTH)
        
        # Use system time to approximate dt
        t_prev = time.time() # Initialized here, updated in update()
        time_state = TimeState(t_current=t_current, t_start=t_start, t_prev=t_prev)

        while (time_state.t_current - time_state.t_start <= END_SIM):
            # Call the update function, passing all necessary current state variables
            time_state, robot_state = await update(
                websocket, time_state, robot_state, controller, ekf, buffers, plot_handles
            )

        await websocket.close()
        print(f"WebSocket connection closed after {END_SIM} seconds. Close plots whenever ready to exit the program.")
        
        # Plotting robot and reference trajectory's x, y, theta 
        plot_x_y_thetha(reference_poses=buffers.ref_poses,
                        reference_heading=buffers.ref_heading,
                        estimated_poses=buffers.est_poses,
                        time_stamps=buffers.time_stamps)

    except websockets.ConnectionClosed:
        print(f"Connection has been closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C). Exiting cleanly.")