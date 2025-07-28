"""Contains EKF algorithm for a differential wheeled robot that is equipped with GPS and IMU.
   References: https://www.cs.jhu.edu/~sleonard/week08.pdf
"""

import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, robot_width):
        # Distance between the center of wheels of the robot
        self.L = robot_width
        
        # Process covariance for states, X [x, y, θ, v, ω]
        # This defines how much we trust our system dynamics
        # Slightly higher values for v and ω because we directly use 
        # control commands to propagate them
        self.Q = np.diag([0.01, 0.01, np.deg2rad(5.0)**2, 0.2, 0.1]) # [m, m, radians, m/s, rad/s]

        # Sensor covariances. Tells the filter how much we should trust the sensors
        # With all 0 inputs we found GPS has around 20cm of noise
        self.R_gps = np.diag([0.2, 0.2]) # [m, m]
        # Took some testing and tuning. But trsuting IMU accelerations 
        # drastically helps with avoiding erratic motions 
        self.R_imu= np.diag([0.175, 0.175, 0.03]) # [m/s^2, m/s^2, rad/s]

    def system_dynamics(self, X: np.ndarray, U: list[float], dt: float)-> np.ndarray:
        """Given the current state, X, and control inputs, U, this method
           defines the process model of the differential wheeled robot.
           We define the states X, as [x, y, θ, v, ω] where (x, y) (in meters), is the 
           position of the robot in the world frame, θ (in radians) is the heading of 
           the robot in the world frame, v (in m/s) is the linear velocity and ω (in rad/s) is 
           the angular velocity of the robot in its body frame. The control inputs, U are 
           [v_left, v_right] (in m/s) which are the left and right wheel linear velocities respectively.
           The process model is derived from the non-linear differential drive kinematics and is 
           linearized for using it in the filter:
                x_k = x_k-1 + dt * v_k-1 * cos(θ_k-1)
                y_k = y_k-1 + dt * v_k-1 * sin(θ_k-1)
                θ_k = θ_k-1 + dt * ω_k-1
                v_k = (v_left + v_right) / 2
                ω_k = (v_right - v_left) / L
           where L is the distance between the wheels (in meters).

           Args: X np.ndarray : The state of the system [x, y, θ, v, ω] (5x1)
                 U list[float] : The control inputs to the system [v_left, v_right] (2x1)
                 dt float      : The time step over which to propagate the state (in seconds)
           
           Returns: X_k np.ndarray : The new predicted state of the system after propagation (5x1)
        """
        x, y, theta, v, omega = X 
        v_left, v_right = U

        # Propagate the state using the differential drive kinematics
        x_k = x + dt * v * np.cos(theta)
        y_k = y + dt * v * np.sin(theta)
        theta_k = theta + dt * omega
        v_k = (v_left + v_right) / 2
        omega_k = (v_right - v_left) / self.L

        X_k = [x_k, y_k, theta_k, v_k, omega_k]
        return np.array(X_k) 
    
    def get_state_transition_A(self, X: np.ndarray, dt: float) -> np.ndarray:
        """Given state X, and dt returns the A matrix 
           This matrix is the process model jacobian w.r.t the states
           This is a 5x5 matrix since we have 5 states [x, y, θ, v, ω].

           Args: X np.ndarray : The state of the system [x, y, θ, v, ω] (5x1)
                 dt float      : The time step over which to propagate the state (in seconds)
           
           Returns: A np.ndarray: The process model Jacobian matrix (5x5)
        """
        
        _, _, theta, v, _ = X
        A = np.array([[1, 0, -dt * v * np.sin(theta), dt * np.cos(theta), 0],
                    [0, 1, dt * v * np.cos(theta), dt * np.sin(theta), 0],
                    [0, 0, 1, 0, dt],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
        return A
    
    def get_gps_obs_model(self, X: np.ndarray) -> np.ndarray:
        """Given state X, defines the observation model for the GPS
           GPS measurements are directly given in meters and are relative to the origin (0, 0).

           Args: X np.ndarray : The state of the system [x, y, θ, v, ω] (5x1)

           Returns: h_gps np.ndaray : The observation model of GPS (2x1)
        """
        # h_gps(X) should predict the GPS readings given the state X
        # Our GPS gives measurement directly in meters so need to use any transformations
        x, y, _, _, _ = X
        h_gps = np.array([x, y])
        return h_gps
    
    def get_gps_jacobian(self) -> np.ndarray:
        """This gives the GPS Observation model jacbobian matrix, H that is computed by takin h_gps derivative 
           w.r.t the states. This is a 2x5 matrix since we have 2 GPS measurements [x, y] and 5 states 
           [x, y, θ, v, ω],

           Returns: H_gps np.ndarray : GPS observation model Jacobian matrix (2x5) 
        """
        H_gps = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]])
        return H_gps
        
    def get_imu_obs_model(self, X: np.ndarray) -> np.ndarray: 
        """Given state X, defines the observation model for the IMU.
           IMU measurements are given as linear accelerations (in m/s^2) and angular 
           velocity (rad/s) in the body frame.

           Args: X np.ndarray : The state of the system [x, y, θ, v, ω] (5x1)

           Returns: h_imu np.ndaray : The observation model of IMU (3x1)
        """
        _, _, _, v, omega = X
        
        # h_imu(X) should predict the IMU readings given the state X
        # We do not have v_dot as one of our states. So use 0 ax acceleration 
        # for the prediction.
        # The EKF will then use the measured ax to update the 'v' state.
        a_x_expected = 0.0 
        # Centripetal acceleration in body frame
        a_y_expected = v * omega 
        # Angular velocity
        omega_expected = omega

        h_imu = np.array([a_x_expected, a_y_expected, omega_expected])
        return h_imu
    
    def get_imu_jacobian(self, X: np.ndarray) -> np.ndarray:
        """This gives the IMU Observation model jacbobian matrix, H that is computed by taking h_imu derivative
           w.r.t the states. This is a 3x5 matrix since we have 3 IMU measurements [ax, ay, omega] and 5 states 
           [x, y, θ, v, ω]

           Args: X np.ndarray : The state of the system [x, y, θ, v, ω] (5x1)

           Returns: h_imu np.ndaray : The observation model of IMU (3x5)   
        """
        _, _, _, v, omega = X
        H_imu = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, omega, v],
                        [0, 0, 0, 0, 1]])
        return H_imu
    
    def prediction_step(self, X: np.ndarray, U: list[float], P: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Given the previous state of the system, X, and previous control commands, U, 
           prediction_step estimates the next state, X_prior and its covariance, P_prior for the
           given time step dt

           Args: X list[float] : Previous state of the system [x, y, θ, v, ω] (5x1)   
                 U list[float] : Previous control inputs to the system [v_left, v_right] (2x1)
                 P np.ndarray  : Previous system covariance mtrix (5x5)
                 dt float      : Current time step in seconds for which the next state is predicted

           Returns: X_prior np.ndarray  : Predicted next state (5x1)
                    P_prior np.ndarray  : Predicted covariance for the next predicted state (5x5)
        """
        X_prior = self.system_dynamics(X, U, dt)
        A = self.get_state_transition_A(X, dt)
        P_prior = A @ P @ A.T + self.Q

        return X_prior, P_prior
    
    def gps_correction_step(self, X_prior: np.ndarray, P_prior: np.ndarray, gps_reading: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Given the prior state estimation, prior state covariance, and a set of measurements from the
           GPS, this method computes a correction step and returns corrected system state and the corresponding
           corrected covariance.

           Args: X_prior np.ndarray  :      Predicted state (5x1)  
                 P_prior np.ndarray  :      Predicted covariance for the state (5x5) 
                 gps_readings list[float] : Current sensor measurement from the GPS

           Returns: X_post np.ndarray : Corrected state (5x1)
                    P_post np.ndarray : Corrected state covariance (5x5) 
        """    
        # Get the number of states to later decide the size of the identity
        # matrix used in the computation
        num_states = len(X_prior)

        H = self.get_gps_jacobian()
        # Residual covariance
        S = H @ P_prior @ H.T + self.R_gps 

        # Filter gain
        K = P_prior @ H.T @ np.linalg.inv(S)
        
        # Get the innovation 
        y = gps_reading - self.get_gps_obs_model(X_prior)
        # Update the state prediction 
        X_post = X_prior + K @ y

        I = np.eye(num_states, num_states)
        # Update the state covariance
        P_post = (I - K @ H) @ P_prior

        return X_post, P_post
    
    def imu_correction_step(self, X_prior: np.ndarray, P_prior: np.ndarray, imu_reading: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Given the prior state estimation, prior state covariance, and a set of measurements from the
           IMU, this method computes a correction step and returns corrected system state and the corresponding
           corrected covariance.

           Args: X_prior np.ndarray  :      Predicted state (5x1)  
                 P_prior np.ndarray  :      Predicted covariance for the state (5x5) 
                 imu_readings list[float] : Current sensor measurement from the IMU (3x1)

           Returns: X_post np.ndarray : Corrected state (5x1)
                    P_post np.ndarray : Corrected state covariance (5x5) 
        """ 
        # Get the number of states to later decide the size of the identity
        # matrix used in the computation
        num_states = len(X_prior)

        H = self.get_imu_jacobian(X_prior) 
        # Residual covariance
        S = H @ P_prior @ H.T + self.R_imu 

        # Filter gain
        K = P_prior @ H.T @ np.linalg.inv(S)

        # Get the innovation 
        y = imu_reading - self.get_imu_obs_model(X_prior)
        
        # Get the innovation 
        X_post = X_prior + K @ y

        I = np.eye(num_states, num_states)
        # Update the state covariance
        sys_covariance_post = (I - K @ H) @ P_prior

        return X_post, sys_covariance_post
    
    def update(self, X: np.ndarray, U: list[float], P: np.ndarray, dt: float, z: list[float], is_gps_updated: bool) -> tuple[np.ndarray, np.ndarray]:
        """
            Performs a full Extended Kalman Filter (EKF) update cycle, including prediction and correction steps,
            using the current state, control inputs, sensor measurements, and time step.

            Args:
                X (np.ndarray): Current state estimate [x, y, θ, v, ω] (5x1)
                U (list[float]): Control inputs [v_left, v_right] (2x1)
                P (np.ndarray): Current state covariance matrix (5x5)
                dt (float): Time step for prediction (in seconds)
                z (list[float]): Sensor measurements, concatenated as [gps_x, gps_y, imu_ax, imu_ay, imu_omega] (5x1)
                is_gps_updated (bool): Flag indicating if a new GPS measurement is available

            Returns:
                X_post (np.ndarray): Updated state estimate after correction (5x1)
                P_post (np.ndarray): Updated state covariance matrix after correction (5x5)
            """
        X_prior, P_prior = self.prediction_step(X, U, P, dt)
        X_post, P_post = X_prior, P_prior
        # GPS doesn't update as regulargly as IMU. Update the state using GPS measurements
        # when a new sensor measurement from the GPS is available.
        # Else we risk running updates on stale information, i.e the robot might not really
        # be at the (x, y) that we think it is. 
        if is_gps_updated:
            X_post, P_post = self.gps_correction_step(X_post, P_post, z[:2])
        X_post, P_post = self.imu_correction_step(X_post, P_post, z[2:])
        # Return corrected predictions of states and covariances
        return X_post, P_post






