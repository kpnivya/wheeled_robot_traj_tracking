"""Contains Dynamic Compensator control for wheeled robot trajectory tracking.
   References: https://asco.lcsr.jhu.edu/docs/EN530_678_S2022/lectures/lecture8.pdf
               https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
"""

import numpy as np

class DynamicCompensator():
    def __init__(self, kp: float, 
                 kd: float, 
                 robot_width: float, 
                 velocity_limits: list[float],
                 acceleration_limit: list[float],
                 initial_u1: float =0.1):
        
        # Proportional and Derivative gains for the control law of the form
        # v = ydd − kp * (yd − yd_ref) − kd * ( y − y_ref)
        self.kp = kp
        self.kd = kd

        # Controller needs to know speeds is physically possible for the robot to achieve
        self.accel_limits = acceleration_limit
        self.vel_limits = velocity_limits

        # Distance between the wheels of the robot
        self.L = robot_width
        # Store the compensator output
        self.u1 = initial_u1
        # Control law is valid only when u1 is not 0. So add a minimum value to avoid singularity
        self.u1_min = 0.03
        # Control-hack: See the spot where it is used in the update for more details
        self.u1_max = 0.75

        # Store previous wheel speeds for acceleration limiting
        self.prev_wheel_speeds = np.array([0.0, 0.0]) 
         

    def constraint_acceleration(self, wheel_speeds: list[float], dt: float) -> list[float]:
        """Given desired wheel speeds applies acceleration constraints to wheel speeds.
           Args: wheel_speeds list[float]: Desired [v_left, v_right] from the controller (2x1)
                 dt : Time step for the controller update 
           
           Returns: constrained_speeds list[float] : Final [v_left, v_right] that we can command
        """
        if dt <= 0:
            return wheel_speeds

        # Calculate desired accelerations
        desired_acc = (wheel_speeds - self.prev_wheel_speeds) / dt
        
        # Limit accelerations to [-1.0, 1.0] m/s²
        clipped_acc = np.clip(desired_acc, self.accel_limits[0], self.accel_limits[1])
        
        # Calculate constrained wheel speeds
        constrained_speeds = self.prev_wheel_speeds + clipped_acc * dt
        
        return constrained_speeds
    

    def update(self, state: np.ndarray, reference: np.ndarray, dt: float) -> list[float]:
        """Update the controller state and compute the next wheel speed commands.
        Args:
            state (np.ndarray): Current robot state [x, y, theta, v, ...].
            reference (np.ndarray): Reference trajectory point [x_ref, y_ref, xd_ref, yd_ref, xdd_ref, ydd_ref].
            dt (float): Time step for the controller update.

        Returns:
            list[float]: Tuple (v_left, v_right) of wheel speed commands after applying speed and acceleration constraints.
        """
        x, y, theta, v, _ = state
        x_ref, y_ref, xd_ref, yd_ref, xdd_ref, ydd_ref = reference
        # Current velocity in world frame
        xd = v * np.cos(theta)
        yd = v * np.sin(theta)

        # Compute errors
        pos_error = np.array([x, y]) - np.array([x_ref, y_ref])
        vel_error = np.array([xd, yd]) - np.array([xd_ref, yd_ref])
        
        # Check stability condition. Calculated using Hurwitz stability criteria
        if not (self.kd >= 2 * np.sqrt(self.kp)):
            print(f"Warning: Gains don't fulfill stability criteria kd > 2*sqrt(kp). Provided kp = {self.kp}, kd = {self.kd}")

        Kp = np.diag([self.kp, self.kp])
        Kd = np.diag([self.kd, self.kd])

        # Virtual inputs (desired accelerations in world frame)
        virtual_in = np.array([xdd_ref, ydd_ref]) - Kp @ pos_error - Kd @ vel_error
        v_1, v_2 = virtual_in

        u1d = v_1 * np.cos(theta) + v_2 * np.sin(theta)
        
        # Integrate to get linear velocity command
        self.u1 = self.u1 + u1d * dt
        # Avoid singularity. Control law is valid only when u1 is not 0
        if abs(self.u1) < self.u1_min:
            self.u1 = self.u1_min if self.u1 >= 0 else -self.u1_min
        # Control-hack: Having no upper limits to u1 would have the robot overshoot on straight paths 
        # which then leads to it missing the turns
        if abs(self.u1) > self.u1_max:
            self.u1 = self.u1_max

        # Compute angular velocity command
        u2 = (v_2 * np.cos(theta) - v_1 * np.sin(theta)) / self.u1
        
        # Convert to wheel speeds
        v_left_desired = self.u1 - (self.L / 2) * u2
        v_right_desired = self.u1 + (self.L / 2) * u2
        
        # Apply speed constraints first
        wheel_speeds = np.array([v_left_desired, v_right_desired])
        speed_constrained = np.clip(wheel_speeds, self.vel_limits[0], self.vel_limits[1])
        
        # Apply acceleration constraints
        final_wheel_speeds = self.constraint_acceleration(speed_constrained, dt)

        # Update stored values for next iteration
        self.prev_wheel_speeds = final_wheel_speeds.copy()

        return final_wheel_speeds[0], final_wheel_speeds[1]


