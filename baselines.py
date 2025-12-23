"""
Baseline methods for comparison with HAF
Implements: BOCPD, HMM, Rolling Mean-Variance
"""

import numpy as np
from scipy.stats import norm
from collections import deque

from scipy.stats import t  # 必须导入 t

class BOCPD:
    """Bayesian Online Change Point Detection (Adams & MacKay 2007)"""
    
    def __init__(self, hazard_rate=0.01, mu0=0, kappa0=1, alpha0=1, beta0=1e-4):
        """
        Args:
            beta0: Scale of the variance prior. 1e-4 is appropriate for daily returns.
        """
        self.hazard = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        self.R = np.array([1.0])
        self.max_runlength = 250
        
        self.mu = [mu0]
        self.kappa = [kappa0]
        self.alpha = [alpha0]
        self.beta = [beta0]
        
        self.change_detected = False
        self.detection_history = []
        
    def update(self, observation):
        x = observation[0] if isinstance(observation, np.ndarray) else observation
        
        T = len(self.R)
        pred_probs = np.zeros(T)
        
        for r in range(T):
            # Student-t predictive parameters
            df = 2 * self.alpha[r]
            mean = self.mu[r]
            scale = np.sqrt(self.beta[r] * (self.kappa[r] + 1) / (self.alpha[r] * self.kappa[r]))
            
            # --- 关键修改开始 ---
            # 使用 t 分布而不是 norm，防止概率下溢为 0
            if scale < 1e-9:
                # 保护：如果 scale 极小，退化为点质量
                pred_probs[r] = 1.0 if abs(x - mean) < 1e-5 else 1e-10
            else:
                pred_probs[r] = t.pdf(x, df, loc=mean, scale=scale)
            # --- 关键修改结束 ---
        
        # Calculate growth probabilities
        R_growth = self.R * pred_probs * (1 - self.hazard)
        
        # Calculate change point probability
        R_cp = np.sum(self.R * pred_probs * self.hazard)
        
        # New run length distribution
        self.R = np.concatenate([[R_cp], R_growth])
        
        # --- 关键修改：归一化保护 ---
        total_prob = np.sum(self.R)
        if total_prob < 1e-12 or np.isnan(total_prob):
            # 如果概率崩了，重置为均匀分布，而不是让它变成 NaN
            self.R = np.ones(len(self.R)) / len(self.R)
        else:
            self.R = self.R / total_prob
        # ------------------------
        
        # Trim
        if len(self.R) > self.max_runlength:
            self.R = self.R[:self.max_runlength]
        
        # Update sufficient statistics
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]
        
        for r in range(min(T, self.max_runlength - 1)):
            kappa_new = self.kappa[r] + 1
            mu_new = (self.kappa[r] * self.mu[r] + x) / kappa_new
            alpha_new = self.alpha[r] + 0.5
            beta_new = self.beta[r] + (self.kappa[r] * (x - self.mu[r])**2) / (2 * kappa_new)
            
            new_mu.append(mu_new)
            new_kappa.append(kappa_new)
            new_alpha.append(alpha_new)
            new_beta.append(beta_new)
        
        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta
        
        # Detect change point
        recent_cp_prob = np.sum(self.R[:5])
        self.change_detected = recent_cp_prob > 0.5
        self.detection_history.append(self.change_detected)
        
        if self.change_detected:
            return 'shift', 0.3
        elif recent_cp_prob > 0.2:
            return 'uncertain', 0.7
        else:
            return 'stable', 1.0
    
    def get_action(self):
        most_likely = np.argmax(self.R)
        if most_likely < len(self.mu):
            return np.array([np.sign(self.mu[most_likely])])
        return np.array([0.0])



class HMM:
    """3-state Hidden Markov Model for regime detection"""
    
    def __init__(self, n_states=3):
        """
        Args:
            n_states: Number of hidden states (regimes)
        """
        self.n_states = n_states
        
        # Transition matrix (bull/bear/neutral)
        self.A = np.array([
            [0.95, 0.03, 0.02],  # Bull -> bull/bear/neutral
            [0.03, 0.95, 0.02],  # Bear -> bull/bear/neutral
            [0.02, 0.02, 0.96]   # Neutral -> bull/bear/neutral
        ])
        
        # Initial state distribution
        self.pi = np.array([0.33, 0.33, 0.34])
        
        # State probabilities (forward algorithm)
        self.alpha = self.pi.copy()
        
        # Emission parameters (mean, std for each state)
        self.means = np.array([0.001, -0.002, 0.0])      # Bull, bear, neutral
        self.stds = np.array([0.01, 0.03, 0.015])
        
        self.regime_history = []
        
    def update(self, observation):
        """Update with new observation using forward algorithm"""
        x = observation[0] if isinstance(observation, np.ndarray) else observation
        
        # Emission probabilities
        B = np.array([norm.pdf(x, self.means[i], self.stds[i]) 
                      for i in range(self.n_states)])
        
        # Forward step
        self.alpha = (self.A.T @ self.alpha) * B
        self.alpha = self.alpha / np.sum(self.alpha)  # Normalize
        
        # Most likely state
        current_state = np.argmax(self.alpha)
        self.regime_history.append(current_state)
        
        # Map to regime names
        regime_map = {0: 'stable', 1: 'shift', 2: 'uncertain'}
        regime = regime_map[current_state]
        
        # Position scale based on state confidence
        confidence = self.alpha[current_state]
        if current_state == 1:  # Bear market
            position_scale = 0.3
        elif confidence < 0.6:  # Low confidence
            position_scale = 0.7
        else:
            position_scale = 1.0
        
        return regime, position_scale
    
    def get_action(self):
        """Get action based on most likely state"""
        current_state = np.argmax(self.alpha)
        if current_state == 0:  # Bull
            return np.array([1.0])
        elif current_state == 1:  # Bear
            return np.array([-0.5])
        else:  # Neutral
            return np.array([0.5])


class RollingMeanVariance:
    """Rolling window mean-variance optimization"""
    
    def __init__(self, window=60, risk_aversion=2.0):
        """
        Args:
            window: Rolling window size
            risk_aversion: Risk aversion parameter
        """
        self.window = window
        self.risk_aversion = risk_aversion
        self.history = deque(maxlen=window)
        
    def update(self, observation):
        """Update with new observation"""
        x = observation[0] if isinstance(observation, np.ndarray) else observation
        self.history.append(x)
        
        # Simple regime detection based on volatility
        if len(self.history) >= 20:
            recent_vol = np.std(list(self.history)[-20:])
            older_vol = np.std(list(self.history)[-40:-20]) if len(self.history) >= 40 else recent_vol
            
            if recent_vol > older_vol * 1.5:
                return 'uncertain', 0.7
            else:
                return 'stable', 1.0
        
        return 'uncertain', 0.8
    
    def get_action(self):
        """Get action based on mean-variance optimization"""
        if len(self.history) < 10:
            return np.array([0.5])
        
        mean_return = np.mean(self.history)
        variance = np.var(self.history)
        
        # Simple mean-variance: w = mu / (lambda * sigma^2)
        if variance > 0:
            weight = mean_return / (self.risk_aversion * variance)
            weight = np.clip(weight, -1, 1)  # Limit to [-1, 1]
        else:
            weight = 0.5
        
        return np.array([weight])


class BayesianOnlineLearning:
    """Bayesian online learning with single Gaussian (no credal sets)"""
    
    def __init__(self, n_assets=1):
        """Simple Bayesian learner"""
        self.mu = np.zeros(n_assets)
        self.sigma = np.eye(n_assets) * 0.01
        self.obs_noise = np.eye(n_assets) * 0.01
        self.n_assets = n_assets
        
    def update(self, observation):
        """Bayesian update"""
        # Kalman filter update
        sigma_inv = np.linalg.inv(self.sigma)
        noise_inv = np.linalg.inv(self.obs_noise)
        
        sigma_new_inv = sigma_inv + noise_inv
        self.sigma = np.linalg.inv(sigma_new_inv)
        self.mu = self.sigma @ (sigma_inv @ self.mu + noise_inv @ observation)
        
        # Simple volatility-based regime detection
        volatility = np.sqrt(np.trace(self.sigma))
        if volatility > 0.02:
            return 'uncertain', 0.7
        else:
            return 'stable', 1.0
    
    def get_action(self):
        """Get action based on posterior mean"""
        if self.n_assets == 1:
            return np.array([np.sign(self.mu[0]) if np.abs(self.mu[0]) > 0.0001 else 0.5])
        else:
            # Normalize
            if np.sum(np.abs(self.mu)) > 0:
                return self.mu / np.sum(np.abs(self.mu))
            return np.ones(self.n_assets) / self.n_assets
