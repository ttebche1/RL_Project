import numpy as np

class ParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.particles = np.random.uniform(low=-1.0, high=1.0, size=(num_particles, 2)) # Particle locations, uniform across the environment
        self.weights = np.ones(num_particles) / num_particles                           # Array of particle probabilities
        self.estimate = np.array([0.0, 0.0])                                            # Estimate of target location
        
    def predict(self):
        """Randomly moves particles"""
        self.particles += np.random.normal(0, 0.01, self.particles.shape)
        self.particles = np.clip(self.particles, -1.0, 1.0) # Keep particles within bounds

    def update(self, agent_position, measured_distance, measurement_noise_std=0.1):
        """Estimate target location"""
        # Compute distance of each particle to the agent
        expected_distances = np.linalg.norm(self.particles - agent_position, axis=1)
        
        # Compute likelihood of each particle
        errors = measured_distance - expected_distances
        likelihoods = np.exp(-0.5 * (errors / measurement_noise_std) ** 2)
        
        # Update weights of each particle (particle with the highest weight is the best)
        self.weights *= likelihoods
        self.weights += 1e-300                  # Avoid zero weights
        self.weights /= np.sum(self.weights)    # Normalize
        
        # Measure how many particles are meaningful
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < self.num_particles / 2:
            self.resample()
            
        # Update estimate of target location
        self.estimate = np.average(self.particles, weights=self.weights, axis=0)
    
    def resample(self):
        """Prune meaningless particles"""
        # Sample particles by their weight (high-weighted particles get selected more than low-weighted ones)
        indices = np.random.choice(
            self.num_particles, 
            size=self.num_particles, 
            p=self.weights
        )

        # Create new particles
        self.particles = self.particles[indices]

        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_estimate(self):
        return self.estimate.copy()
    
    def get_uncertainty(self):
        """Return covariance of particle distribution"""
        weighted_mean = self.estimate
        diff = self.particles - weighted_mean
        covariance = np.average(
            diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
            weights=self.weights, axis=0
        )
        return covariance