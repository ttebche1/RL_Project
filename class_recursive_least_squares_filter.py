# Recursive least squares filter for static target localization from range measurements

import numpy as np

class RecursiveLeastSquaresFilter:
    def __init__(self, forgetting_factor=0.98, initial_uncertainty=1000.0):
        self.lambda_ = forgetting_factor                            # How much to forget old measurements (1 = no forgetting)
        self.lambda_inv = 1.0 / forgetting_factor        
        self.est_target_loc_vec = np.zeros(2, dtype=np.float64)     # Estimated target location
        self.P = initial_uncertainty * np.eye(2, dtype=np.float64)  # Covariance matrix
        
    def update(self, agent_loc_vec, meas_dist_to_target_mag):
        """Update target location with single range measurement from one sensor"""
        # Current estimated distance to target
        est_dist_to_target_vec = self.est_target_loc_vec - agent_loc_vec
        est_dist_to_target_mag = np.linalg.norm(est_dist_to_target_vec)
        
        # Compute Jacobian H
        if est_dist_to_target_mag < 1e-6:
            H = np.array([1.0, 0.0], dtype=np.float64)          # Avoid division by zero
        else:
            H = est_dist_to_target_vec / est_dist_to_target_mag # Normalize distance to target vector
        
        # Compute filter gain
        PH = self.P @ H
        K = PH / (self.lambda_ + H @ PH)
        
        # Update target location estimate
        est_dist_to_target_mag_error = meas_dist_to_target_mag - est_dist_to_target_mag
        self.est_target_loc_vec += K * est_dist_to_target_mag_error
        
        # Update covariance
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = (I_KH @ self.P @ I_KH.T + np.outer(K, K) * (self.lambda_ - 1)) * self.lambda_inv
        
        return self.est_target_loc_vec.copy()
    
    def get_target_loc(self):
        """Get current target location estimate"""
        return self.est_target_loc_vec.copy()
    
    def get_uncertainty(self):
        """Get position uncertainty (covariance matrix)"""
        return self.P.copy()