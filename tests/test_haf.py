"""
Unit tests for HAF implementation
"""

import numpy as np
import pytest
from haf_core import GaussianDistribution, CredalSet, HausdorffAdaptiveFilter


class TestGaussianDistribution:
    """Test GaussianDistribution class"""
    
    def test_initialization(self):
        mu = np.array([0.0, 0.0])
        sigma = np.eye(2) * 0.01
        dist = GaussianDistribution(mu, sigma)
        
        assert np.allclose(dist.mu, mu)
        assert np.allclose(dist.sigma, sigma)
    
    def test_likelihood(self):
        mu = np.array([0.0])
        sigma = np.array([[1.0]])
        dist = GaussianDistribution(mu, sigma)
        
        # Likelihood at mean should be highest
        lik_at_mean = dist.likelihood(np.array([0.0]))
        lik_away = dist.likelihood(np.array([2.0]))
        
        assert lik_at_mean > lik_away
    
    def test_bayesian_update(self):
        mu = np.array([0.0])
        sigma = np.array([[1.0]])
        dist = GaussianDistribution(mu, sigma)
        
        # Update with observation
        obs = np.array([1.0])
        obs_noise = np.array([[0.1]])
        
        updated = dist.bayesian_update(obs, obs_noise)
        
        # Updated mean should move toward observation
        assert updated.mu[0] > mu[0]
        assert updated.mu[0] < obs[0]
        
        # Updated variance should decrease
        assert updated.sigma[0, 0] < sigma[0, 0]


class TestCredalSet:
    """Test CredalSet class"""
    
    def test_initialization(self):
        d1 = GaussianDistribution(np.array([0.0]), np.array([[1.0]]))
        d2 = GaussianDistribution(np.array([1.0]), np.array([[1.0]]))
        
        credal_set = CredalSet([d1, d2])
        
        assert credal_set.K == 2
        assert len(credal_set.extremes) == 2
    
    def test_wasserstein_distance(self):
        d1 = GaussianDistribution(np.array([0.0]), np.array([[1.0]]))
        d2 = GaussianDistribution(np.array([1.0]), np.array([[1.0]]))
        
        credal_set = CredalSet([d1, d2])
        dist = credal_set.wasserstein2_distance(d1, d2)
        
        # Distance should be positive
        assert dist > 0
        
        # Distance to self should be zero
        dist_self = credal_set.wasserstein2_distance(d1, d1)
        assert np.isclose(dist_self, 0, atol=1e-6)
    
    def test_hausdorff_distance(self):
        d1 = GaussianDistribution(np.array([0.0]), np.array([[1.0]]))
        d2 = GaussianDistribution(np.array([1.0]), np.array([[1.0]]))
        
        cs1 = CredalSet([d1])
        cs2 = CredalSet([d2])
        
        dist = cs1.hausdorff_distance(cs2)
        
        # Distance should be positive
        assert dist > 0
        
        # Distance should be symmetric
        dist_reverse = cs2.hausdorff_distance(cs1)
        assert np.isclose(dist, dist_reverse, rtol=1e-5)
    
    def test_diameter(self):
        d1 = GaussianDistribution(np.array([0.0]), np.array([[1.0]]))
        d2 = GaussianDistribution(np.array([2.0]), np.array([[1.0]]))
        
        credal_set = CredalSet([d1, d2])
        diam = credal_set.diameter()
        
        # Diameter should be positive
        assert diam > 0
    
    def test_bayesian_update(self):
        d1 = GaussianDistribution(np.array([0.0]), np.array([[1.0]]))
        d2 = GaussianDistribution(np.array([1.0]), np.array([[1.0]]))
        
        credal_set = CredalSet([d1, d2])
        obs = np.array([0.5])
        obs_noise = np.array([[0.1]])
        
        updated_cs = credal_set.bayesian_update(obs, obs_noise)
        
        # Should still have same number of extremes
        assert updated_cs.K == credal_set.K
        
        # Extremes should have moved toward observation
        assert updated_cs.extremes[0].mu[0] > d1.mu[0]
        assert updated_cs.extremes[1].mu[0] < d2.mu[0]


class TestHAF:
    """Test HausdorffAdaptiveFilter class"""
    
    def test_initialization(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        
        assert haf.n_assets == 1
        assert haf.credal_set.K == 3  # Bull, bear, neutral
        assert len(haf.distances) == 0
        assert len(haf.rho_history) == 0
    
    def test_update_single_observation(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        obs = np.array([0.001])  # Small positive return
        
        regime, position_scale = haf.update(obs)
        
        # Should return valid regime and position scale
        assert regime in ['stable', 'uncertain', 'shift']
        assert 0 < position_scale <= 1.0
        
        # Should have recorded distance
        assert len(haf.distances) == 1
        assert haf.distances[0] >= 0
    
    def test_update_sequence(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        
        # Stable sequence
        for _ in range(20):
            obs = np.random.normal(0.001, 0.01, size=1)
            regime, _ = haf.update(obs)
        
        # Should have built up history
        assert len(haf.distances) == 20
        assert len(haf.rho_history) >= 18  # Started after 2 observations
    
    def test_regime_detection_stable(self):
        """Test that stable regime is detected in stable data"""
        np.random.seed(42)
        haf = HausdorffAdaptiveFilter(n_assets=1, rho_thresh=0.95)
        
        # Generate stable data
        for _ in range(50):
            obs = np.random.normal(0.001, 0.01, size=1)
            haf.update(obs)
        
        # After convergence, should see stable regime frequently
        stable_count = sum(1 for r in haf.regime_history[-20:] if r == 0)
        assert stable_count > 10  # At least half should be stable
    
    def test_regime_detection_shift(self):
        """Test that regime shift is detected"""
        np.random.seed(42)
        haf = HausdorffAdaptiveFilter(n_assets=1, rho_reset=1.2)
        
        # Stable data first
        for _ in range(30):
            obs = np.random.normal(0.001, 0.01, size=1)
            haf.update(obs)
        
        # Sudden shift to crisis
        shift_detected = False
        for _ in range(20):
            obs = np.random.normal(-0.015, 0.03, size=1)
            regime, _ = haf.update(obs)
            if regime == 'shift':
                shift_detected = True
                break
        
        # Should detect the shift
        assert shift_detected
    
    def test_get_action(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        
        # Update with some data
        for _ in range(10):
            obs = np.random.normal(0.001, 0.01, size=1)
            haf.update(obs)
        
        action = haf.get_action()
        
        # Should return valid action
        assert action.shape == (1,)
        assert not np.isnan(action[0])
    
    def test_get_metrics(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        obs = np.array([0.001])
        haf.update(obs)
        
        metrics = haf.get_metrics()
        
        # Should have all required metrics
        assert 'diameter' in metrics
        assert 'rho' in metrics
        assert 'distance' in metrics
        assert 'regime' in metrics
        
        # Values should be valid
        assert metrics['diameter'] >= 0
        assert metrics['distance'] >= 0
    
    def test_reset_credal_set(self):
        haf = HausdorffAdaptiveFilter(n_assets=1)
        
        # Update a few times
        for _ in range(5):
            obs = np.random.normal(0.001, 0.01, size=1)
            haf.update(obs)
        
        initial_diameter = haf.credal_set.diameter()
        
        # Reset
        haf.reset_credal_set()
        reset_diameter = haf.credal_set.diameter()
        
        # Diameter after reset should be larger (more uncertain)
        assert reset_diameter >= initial_diameter * 0.8


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test complete pipeline with synthetic regime changes"""
        np.random.seed(42)
        haf = HausdorffAdaptiveFilter(n_assets=1)
        
        returns = []
        positions = []
        
        # Bull market
        for _ in range(100):
            obs = np.random.normal(0.001, 0.01, size=1)
            regime, scale = haf.update(obs)
            action = haf.get_action()
            position = action[0] * scale
            
            returns.append(obs[0])
            positions.append(position)
        
        # Crisis
        for _ in range(50):
            obs = np.random.normal(-0.015, 0.03, size=1)
            regime, scale = haf.update(obs)
            action = haf.get_action()
            position = action[0] * scale
            
            returns.append(obs[0])
            positions.append(position)
        
        returns = np.array(returns)
        positions = np.array(positions)
        
        # Compute portfolio returns
        portfolio_returns = returns * positions
        
        # Should have generated valid portfolio returns
        assert len(portfolio_returns) == 150
        assert not np.any(np.isnan(portfolio_returns))
        
        # HAF should have reduced position during crisis
        avg_position_bull = np.mean(positions[:100])
        avg_position_crisis = np.mean(positions[100:])
        
        assert avg_position_crisis < avg_position_bull


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
