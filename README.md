# Trajectory Tracking for Differential Wheeled Robot using EKF and Feedback Control

## Objective
A lightweight Python project that simulates a differential wheeled robot, using only open-source Python packages. The project contains code for having a differential wheeled robot track a pre-defined reference trajectory. 

## System Requirements 
Conda is required to run the sciprt. Install conda for your system if needed from: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html .\
The code was developed and tested on Python 3.10.18. To avoid any issues use Python 3.10.x or higher.

## Installation 
1. Create a new conda environment 
```
conda create -n diff-robot-env python=3.10.18 -y
```
2. Activate the environment
```
conda activate diff-robot-env
```
3. Clone the repository 
```
git clone https://github.com/kpnivya/wheeled_robot_traj_tracking.git
```
```
cd wheeled_robot_traj_tracking
```
4. Install dependencies 
```
pip install --upgrade pip
```
```
pip install -r requirements.txt
```
4. Run the simulation!
```
python3 src/main.py
```

## Package Info

- `src` 
Contains all the source code to run the simulation. 
    - `main.py`
        Runs the simulated robot by keeping track of time and running controller and EKF updates at each time step. The simulation by default runs for 20 seconds in system time. User facing variables are defined under the `Const variables and helper classes` section of this file. You can set acceleration and velocity limits of the robot, change the robot's width, robot's initial conditions and the reference trajectory that the robot needs to follow using these variables. This file automatically collects data during the simulation and shows a live-plot that contains the robot's path, reference trajectory, and GPS path. At the end of 20 seconds, the program should automatically show plots for Robot Heading vs Reference Trajectory Heading, Robot X pose vs Reference Trajectory X pose, Robot Y pose vs Reference trajectory Y pose. These plots are provided to give an idea about how well the robot tracks the reference trajectory.
    - `controller.py`
        The reference trajectory is tracked using a controller based on dynamic feedback linearization. This file contains the `DynamicCompensator` class that includes the controller `update` method. The controller takes into consideration the robot width, its maximum acceleration and velocity.
    - `ekf.py`
        The robot has two sensors, a GPS that gives the robot position in world origin and an IMU that gives the robot acceleration and angular velcoity in its frame. These sensors can be used to estimate the state of the robot. So this file contains `ExtendedKalmanFilter` class that includes the process model for the system dynamics, the observation models for the sensors, and the EKF algorithm. The class's `update` method runs a prediction and correction step to estimate the current state of the robot. Covariances are set based on experimentation with the sensors and are initialized in the constructor.  
    - `helpers.py`
        Contains WebSocket utitilies to connect to the server, JSON utilites to validate messages and convert them to and from JSON. Also contains helper functions that are used to display simulation plots.
    - `reference_traj.py`
        Contains simple trajectories you can use to test your robot's trajectory tracking capability. Options are `course-8` (the default option and the one that the serves gives a L2 score for), `straight-line` and `semi-circle`.
