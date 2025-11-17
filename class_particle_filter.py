import numpy as np
import torch

class ParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()
    
    def reset(self):
        """Reset particle filter"""
        self.particles =torch.rand(self.num_particles, 2, device=self.device) * 2 - 1           # Particle locations, uniform across the environment
        self.weights = torch.ones(self.num_particles, device=self.device) / self.num_particles  # Array of particle probabilities
        self.estimate = torch.zeros(2, device=self.device)                                      # Estimate of target location
        
    def predict(self):
        """Randomly moves particles"""
        self.particles += torch.randn_like(self.particles) * 0.05
        self.particles = torch.clamp(self.particles, -1.0, 1.0) # Keep particles within bounds
        self.particles[torch.isnan(self.particles)] = 0.0
        self.particles[torch.isinf(self.particles)] = 0.0

    def update(self, agent_position, measured_distance, measurement_noise_std=0.1):
        """Estimate target location"""
        # Compute distance of each particle to the agent
        agent_position = torch.tensor(agent_position, dtype=torch.float32, device=self.device)
        expected_distances = torch.norm(self.particles - agent_position, dim=1)
        
        # Compute likelihood of each particle
        errors = measured_distance - expected_distances
        likelihoods = torch.exp(-0.5 * (errors / measurement_noise_std) ** 2) / \
            (measurement_noise_std * (2 * torch.pi) ** 0.5)

        # Update weights of each particle (particle with the highest weight is the best)
        self.weights *= likelihoods
        self.weights += 1e-300                  # Avoid zero weights
        self.weights /= torch.sum(self.weights) # Normalize
        
        # Measure how many particles are meaningful
        neff = 1.0 / torch.sum(self.weights ** 2)
        if neff < self.num_particles / 2:
            self.resample()
            
        # Update estimate of target location
        self.estimate = torch.sum(self.particles * self.weights.unsqueeze(1), dim=0)
        if torch.isnan(self.estimate).any() or torch.isinf(self.estimate).any():
            self.estimate = torch.zeros_like(self.estimate)
    
    def resample(self):
        """Prune meaningless particles"""
        # Create evenly spaced particle positions with some noise
        positions = (torch.arange(self.num_particles, device=self.device).float() \
                     + torch.rand(1, device=self.device)) / self.num_particles
        positions = torch.clamp(positions, 0.0, 1.0 - 1e-8)

        # Sample particles based on their weights
        cdf = torch.cumsum(self.weights, dim=0)
        cdf[-1] = 1.0
        indices = torch.searchsorted(cdf, positions)
        indices = torch.clamp(indices, 0, self.num_particles - 1)

        # Create new particles
        self.particles = self.particles[indices]
        self.particles += torch.randn_like(self.particles) * 0.05
        self.particles[torch.isnan(self.particles)] = 0.0
        self.particles[torch.isinf(self.particles)] = 0.0

        # Reset weights
        self.weights.fill_(1.0 / self.num_particles)
    
    def get_estimate(self):
        return self.estimate.clone() 
    
    def get_uncertainty(self):
        """Return covariance of particle distribution"""
        weighted_mean = self.estimate
        diff = self.particles - weighted_mean
        covariance = torch.einsum('ni,nj->ij', diff * self.weights.unsqueeze(1), diff)
        return covariance