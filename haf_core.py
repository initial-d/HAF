"""
Hausdorff-Adaptive Filter (HAF) Core Implementation
Based on "Robust Online Learning in Non-Stationary Markets: A Credal Set Approach"
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from typing import List, Tuple, Optional


class GaussianDistribution:
    """Single Gaussian distribution representing a market regime"""
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu.copy()
        self.sigma = sigma.copy()
        
    def copy(self):
        return GaussianDistribution(self.mu, self.sigma)
    
    def likelihood(self, x: np.ndarray) -> float:
        """Compute likelihood of observation x"""
        return multivariate_normal.pdf(x, mean=self.mu, cov=self.sigma)
    
    def bayesian_update(self, x: np.ndarray, obs_noise: np.ndarray):
        """Bayesian update given observation x"""
        sigma_inv = np.linalg.inv(self.sigma)
        noise_inv = np.linalg.inv(obs_noise)
        
        # Update covariance
        sigma_new_inv = sigma_inv + noise_inv
        sigma_new = np.linalg.inv(sigma_new_inv)
        
        # Update mean
        mu_new = sigma_new @ (sigma_inv @ self.mu + noise_inv @ x)
        
        return GaussianDistribution(mu_new, sigma_new)


class CredalSet:
    """Credal set represented as convex hull of extreme distributions"""
    
    def __init__(self, extremes: List[GaussianDistribution]):
        self.extremes = [e.copy() for e in extremes]
        self.K = len(extremes)
    
    def copy(self):
        return CredalSet(self.extremes)
    
    def wasserstein2_distance(self, p1: GaussianDistribution, 
                             p2: GaussianDistribution) -> float:
        """Wasserstein-2 distance between two Gaussians"""
        # ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})
        mu_diff = p1.mu - p2.mu
        mu_term = np.dot(mu_diff, mu_diff)
        
        # Matrix square root computation
        sqrt_sigma2 = sqrtm(p2.sigma)
        M = sqrt_sigma2 @ p1.sigma @ sqrt_sigma2
        sqrt_M = sqrtm(M)
        
        sigma_term = np.trace(p1.sigma + p2.sigma - 2 * sqrt_M)
        
        return np.sqrt(mu_term + sigma_term)
    
    def hausdorff_distance(self, other: 'CredalSet') -> float:
        """Compute Hausdorff distance between two credal sets"""
        # max(max_i min_j d(Pi, Qj), max_j min_i d(Pi, Qj))
        
        # Direction 1: for each P in self, find closest Q in other
        max_dist_1 = 0
        for p in self.extremes:
            min_dist = float('inf')
            for q in other.extremes:
                dist = self.wasserstein2_distance(p, q)
                min_dist = min(min_dist, dist)
            max_dist_1 = max(max_dist_1, min_dist)
        
        # Direction 2: for each Q in other, find closest P in self
        max_dist_2 = 0
        for q in other.extremes:
            min_dist = float('inf')
            for p in self.extremes:
                dist = self.wasserstein2_distance(p, q)
                min_dist = min(min_dist, dist)
            max_dist_2 = max(max_dist_2, min_dist)
        
        return max(max_dist_1, max_dist_2)
    
    def diameter(self) -> float:
        """Compute diameter of credal set (max distance between extremes)"""
        max_dist = 0
        for i, p1 in enumerate(self.extremes):
            for p2 in self.extremes[i+1:]:
                dist = self.wasserstein2_distance(p1, p2)
                max_dist = max(max_dist, dist)
        return max_dist
    
    def bayesian_update(self, x: np.ndarray, obs_noise: np.ndarray):
        """Update all extreme distributions"""
        new_extremes = [p.bayesian_update(x, obs_noise) for p in self.extremes]
        return CredalSet(new_extremes)
    
    def pessimistic_mean(self) -> np.ndarray:
        """Return mean of most pessimistic (lowest expected return) distribution"""
        min_return = float('inf')
        pessimistic_mu = None
        
        for dist in self.extremes:
            expected_return = np.mean(dist.mu)  # Simple: average of returns
            if expected_return < min_return:
                min_return = expected_return
                pessimistic_mu = dist.mu
        
        return pessimistic_mu


class HausdorffAdaptiveFilter:
    """
    Hausdorff-Adaptive Filter (HAF) for regime detection and online learning
    """
    
    def __init__(self, 
                 n_assets: int = 1,
                 rho_thresh: float = 0.95,
                 rho_reset: float = 1.2,
                 window: int = 10,
                 safety_factor: float = 0.3):
        """
        Args:
            n_assets: Number of assets
            rho_thresh: Threshold for stable regime detection
            rho_reset: Threshold for regime shift detection
            window: Window for moving average of contraction ratio
            safety_factor: Position scaling during regime shift
        """
        self.n_assets = n_assets
        self.rho_thresh = rho_thresh
        self.rho_reset = rho_reset
        self.window = window
        self.safety_factor = safety_factor
        
        # Initialize credal set with 3 extreme distributions
        self.credal_set = self._initialize_credal_set()
        
        # History tracking
        self.distances = []
        self.rho_history = []
        self.diameter_history = []
        self.regime_history = []  # 0: stable, 1: uncertain, 2: shift detected
        
        # Observation noise (assumed known or estimated)
        self.obs_noise = np.eye(n_assets) * 0.01
        
    def _initialize_credal_set(self) -> CredalSet:
        """Initialize credal set with bull, bear, neutral regimes"""
        # Bull market: positive mean, low volatility
        bull = GaussianDistribution(
            mu=np.ones(self.n_assets) * 0.001,
            sigma=np.eye(self.n_assets) * 0.0001
        )
        
        # Bear market: negative mean, high volatility
        bear = GaussianDistribution(
            mu=np.ones(self.n_assets) * -0.002,
            sigma=np.eye(self.n_assets) * 0.0009
        )
        
        # Neutral market: zero mean, medium volatility
        neutral = GaussianDistribution(
            mu=np.zeros(self.n_assets),
            sigma=np.eye(self.n_assets) * 0.0004
        )
        
        return CredalSet([bull, bear, neutral])
    
    def reset_credal_set(self):
        """Reset to broad, uninformative prior"""
        self.credal_set = self._initialize_credal_set()
    
    def update(self, observation: np.ndarray) -> Tuple[str, float]:
        """
        Update filter with new observation
        
        Args:
            observation: Market observation (returns)
            
        Returns:
            regime: 'stable', 'uncertain', or 'shift'
            position_scale: Recommended position scaling factor
        """
        # Store previous credal set
        prev_credal_set = self.credal_set.copy()
        
        # Bayesian update
        self.credal_set = self.credal_set.bayesian_update(observation, self.obs_noise)
        
        # Compute Hausdorff distance
        d_t = self.credal_set.hausdorff_distance(prev_credal_set)
        self.distances.append(d_t)
        
        # Compute diameter (epistemic uncertainty)
        diameter = self.credal_set.diameter()
        self.diameter_history.append(diameter)
        
        # Compute contraction ratio (if enough history)
        regime = 'uncertain'
        position_scale = 1.0
        
        if len(self.distances) >= 2:
            rho_t = self.distances[-1] / (self.distances[-2] + 1e-10)
            self.rho_history.append(rho_t)
            
            # Moving average of rho
            if len(self.rho_history) >= self.window:
                rho_bar = np.mean(self.rho_history[-self.window:])
            else:
                rho_bar = np.mean(self.rho_history)
            
            # Regime detection
            if rho_bar > self.rho_reset:
                # Regime shift detected
                regime = 'shift'
                self.reset_credal_set()
                position_scale = self.safety_factor
                self.regime_history.append(2)
            elif rho_bar < self.rho_thresh:
                # Stable regime
                regime = 'stable'
                position_scale = 1.0
                self.regime_history.append(0)
            else:
                # Uncertain regime
                regime = 'uncertain'
                # Scale position inversely with epistemic uncertainty
                position_scale = 1.0 / (1.0 + diameter)
                self.regime_history.append(1)
        else:
            self.regime_history.append(1)
        
        return regime, position_scale
    
    def get_action(self, risk_aversion: float = 2.0) -> np.ndarray:
        """
        Compute pessimistic action (portfolio weights)
        
        Args:
            risk_aversion: Risk aversion parameter
            
        Returns:
            Portfolio weights
        """
        # Use pessimistic mean estimate
        mu_pessimistic = self.credal_set.pessimistic_mean()
        
        # Simple mean-variance optimization with pessimistic estimate
        # w = (1/lambda) * Sigma^{-1} * mu (assuming single-period)
        
        # Use average covariance from extreme distributions
        avg_sigma = np.mean([dist.sigma for dist in self.credal_set.extremes], axis=0)
        
        try:
            sigma_inv = np.linalg.inv(avg_sigma)
            weights = (1.0 / risk_aversion) * sigma_inv @ mu_pessimistic
            
            # Normalize to sum to 1 (for long-only portfolio)
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
            else:
                weights = np.ones(self.n_assets) / self.n_assets
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def get_metrics(self) -> dict:
        """Get current metrics for monitoring"""
        return {
            'diameter': self.diameter_history[-1] if self.diameter_history else 0,
            'rho': self.rho_history[-1] if self.rho_history else 0,
            'distance': self.distances[-1] if self.distances else 0,
            'regime': self.regime_history[-1] if self.regime_history else 1
        }
