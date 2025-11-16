import numpy as np

class LeastSquaresFilter:
    def __init__(self, initial_estimate=np.zeros(2), initial_covariance=1e3*np.eye(2)):
        self.estimate = np.array(initial_estimate, dtype=float) # Estimate of target location
        self.cov = np.array(initial_covariance, dtype=float)    # Covariance (uncertainty) of estimate

    def update(self, agent_loc_vec, dist_measurement, measurement_std=0.1):
        """
        Update the estimate of the target location based on a new distance measurement

        Args:
            agent_loc_vec (np.ndarray): Position of the agent
            distance_measurement (float): Measured distance to the target
            measurement_std (float): Standard deviation of the measurement noise
        
        Returns:
            np.ndarray: Updated estimate of the target location
        """
        # Compute distance between estimate and agent position
        dist_to_target_est_vec = self.estimate - agent_loc_vec
        dist_to_target_est_mag = np.linalg.norm(dist_to_target_est_vec)
        
        # Linearize distance to target estimation
        H = dist_to_target_est_vec / dist_to_target_est_mag if dist_to_target_est_mag > 1e-6 else np.zeros(2)

        # Compute error between the actual and etimated distance measurements
        dist_est_error = dist_measurement - dist_to_target_est_mag

        # Compute actual distance measurement covariance
        R = measurement_std ** 2
        cov_H = self.cov @ H
        S = H @ cov_H + R

        # Compute gain
        if S > 1e-12:
            K = cov_H / S
        else:
            K = np.zeros(2)

        # Update estimate
        self.estimate = self.estimate + K * dist_est_error
        self.cov = (np.eye(2) - np.outer(K, H)) @ self.cov @ (np.eye(2) - np.outer(K, H)).T + np.outer(K, K) * R

        return self.estimate