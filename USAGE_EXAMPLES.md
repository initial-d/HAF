# Usage Examples

This document provides practical code examples for common use cases.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Custom Parameters](#custom-parameters)
3. [Real-Time Monitoring](#real-time-monitoring)
4. [Multi-Asset Portfolio](#multi-asset-portfolio)
5. [Backtesting](#backtesting)
6. [Integration Patterns](#integration-patterns)

---

## Basic Usage

### Minimal Example (10 lines)

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

# Initialize
haf = HausdorffAdaptiveFilter(n_assets=1)

# Process observations
returns = np.random.normal(0.001, 0.01, size=(100, 1))
for ret in returns:
    regime, scale = haf.update(ret)
    action = haf.get_action()
    print(f"Regime: {regime}, Position: {action[0]*scale:.3f}")
```

### Complete Example with Metrics

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

# Setup
haf = HausdorffAdaptiveFilter(n_assets=1)
positions = []
regimes = []

# Simulate trading
for t in range(200):
    # Generate synthetic return
    ret = np.random.normal(0.001, 0.01, size=1)
    
    # HAF update
    regime, position_scale = haf.update(ret)
    action = haf.get_action()
    final_position = action[0] * position_scale
    
    # Track results
    positions.append(final_position)
    regimes.append(regime)
    
    # Get detailed metrics
    metrics = haf.get_metrics()
    
    if t % 50 == 0:
        print(f"\nTimestep {t}:")
        print(f"  Regime: {regime}")
        print(f"  Position: {final_position:.3f}")
        print(f"  Uncertainty (diameter): {metrics['diameter']:.4f}")
        print(f"  Contraction ratio: {metrics['rho']:.4f}")

# Summary
print(f"\nRegime distribution:")
print(f"  Stable: {regimes.count('stable')} periods")
print(f"  Uncertain: {regimes.count('uncertain')} periods")
print(f"  Shifts: {regimes.count('shift')} periods")
```

---

## Custom Parameters

### Conservative Strategy

```python
# Less sensitive to regime shifts, more stable positions
haf = HausdorffAdaptiveFilter(
    n_assets=1,
    rho_thresh=0.90,        # Detect uncertainty earlier
    rho_reset=1.50,         # Higher threshold for regime shifts
    window=20,              # Longer smoothing window
    safety_factor=0.5       # Less aggressive position reduction
)
```

### Aggressive Strategy

```python
# More sensitive, faster adaptation
haf = HausdorffAdaptiveFilter(
    n_assets=1,
    rho_thresh=0.98,        # Only flag clear uncertainty
    rho_reset=1.10,         # Lower threshold, faster detection
    window=5,               # Shorter window, faster response
    safety_factor=0.2       # More aggressive reduction on shifts
)
```

### High-Frequency Adapted

```python
# For higher frequency data (e.g., hourly)
haf = HausdorffAdaptiveFilter(
    n_assets=1,
    rho_thresh=0.95,
    rho_reset=1.15,         # Slightly lower for more frequent data
    window=30,              # Longer window to smooth noise
    safety_factor=0.3
)

# Scale observation noise for frequency
haf.obs_noise = np.eye(1) * 0.0001  # Smaller noise for hourly data
```

---

## Real-Time Monitoring

### Dashboard-Style Monitoring

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter
from datetime import datetime

class HAFMonitor:
    def __init__(self, haf):
        self.haf = haf
        self.alerts = []
    
    def process_tick(self, timestamp, return_data):
        """Process single market tick with monitoring"""
        regime, scale = self.haf.update(return_data)
        action = self.haf.get_action()
        metrics = self.haf.get_metrics()
        
        # Generate alerts
        if regime == 'shift':
            self.alerts.append(f"[{timestamp}] REGIME SHIFT DETECTED!")
        
        if metrics['diameter'] > 0.05:
            self.alerts.append(f"[{timestamp}] High uncertainty: {metrics['diameter']:.4f}")
        
        # Return formatted status
        return {
            'timestamp': timestamp,
            'regime': regime,
            'position': action[0] * scale,
            'uncertainty': metrics['diameter'],
            'rho': metrics['rho'],
            'alert': self.alerts[-1] if self.alerts else None
        }

# Usage
haf = HausdorffAdaptiveFilter(n_assets=1)
monitor = HAFMonitor(haf)

# Simulate real-time feed
for t in range(100):
    ts = datetime.now()
    ret = np.random.normal(0.001, 0.01, size=1)
    status = monitor.process_tick(ts, ret)
    
    if status['alert']:
        print(f"🚨 {status['alert']}")
    
    if t % 10 == 0:
        print(f"[{ts}] Regime: {status['regime']:10s} | "
              f"Position: {status['position']:.3f} | "
              f"Uncertainty: {status['uncertainty']:.4f}")
```

### Logging to File

```python
import numpy as np
import json
from haf_core import HausdorffAdaptiveFilter
from datetime import datetime

# Setup
haf = HausdorffAdaptiveFilter(n_assets=1)
log_file = open('haf_log.jsonl', 'w')

# Trading loop with logging
for t in range(200):
    ret = np.random.normal(0.001, 0.01, size=1)
    regime, scale = haf.update(ret)
    action = haf.get_action()
    metrics = haf.get_metrics()
    
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'timestep': t,
        'return': float(ret[0]),
        'regime': regime,
        'position': float(action[0] * scale),
        'metrics': {
            'diameter': float(metrics['diameter']),
            'rho': float(metrics['rho']),
            'distance': float(metrics['distance'])
        }
    }
    
    # Write to file
    log_file.write(json.dumps(log_entry) + '\n')

log_file.close()
print("Logged 200 timesteps to haf_log.jsonl")
```

---

## Multi-Asset Portfolio

### Simple Multi-Asset

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

# 5 asset portfolio
n_assets = 5
haf = HausdorffAdaptiveFilter(n_assets=n_assets)

# Process multi-asset returns
for t in range(100):
    # Returns for all assets (could be correlated)
    returns = np.random.normal(0.001, 0.01, size=n_assets)
    
    regime, scale = haf.update(returns)
    weights = haf.get_action()  # Shape: (n_assets,)
    
    # Scale all weights
    final_weights = weights * scale
    
    print(f"t={t}: {', '.join([f'w{i}={w:.2f}' for i, w in enumerate(final_weights)])}")
```

### With Correlation Structure

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

n_assets = 3
haf = HausdorffAdaptiveFilter(n_assets=n_assets)

# Define correlation matrix
corr = np.array([[1.0, 0.6, 0.3],
                 [0.6, 1.0, 0.4],
                 [0.3, 0.4, 1.0]])

mean = np.array([0.001, 0.0008, 0.0012])
vol = np.array([0.01, 0.012, 0.009])

# Covariance matrix
cov = np.outer(vol, vol) * corr

portfolio_values = [1.0]

for t in range(200):
    # Generate correlated returns
    returns = np.random.multivariate_normal(mean, cov)
    
    regime, scale = haf.update(returns)
    weights = haf.get_action() * scale
    
    # Portfolio return
    port_return = np.dot(weights, returns)
    portfolio_values.append(portfolio_values[-1] * (1 + port_return))
    
    if (t + 1) % 50 == 0:
        print(f"t={t+1}: Portfolio value = ${portfolio_values[-1]:.2f}")
```

---

## Backtesting

### Simple Backtest

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

def backtest(returns, initial_capital=10000):
    """
    Backtest HAF on historical returns
    
    Args:
        returns: Array of shape (T, n_assets)
        initial_capital: Starting capital
    
    Returns:
        dict with performance metrics
    """
    haf = HausdorffAdaptiveFilter(n_assets=returns.shape[1])
    
    portfolio_values = [initial_capital]
    positions_history = []
    regimes_history = []
    
    for ret in returns:
        # HAF decision
        regime, scale = haf.update(ret)
        weights = haf.get_action() * scale
        
        # Portfolio return
        port_return = np.dot(weights, ret)
        new_value = portfolio_values[-1] * (1 + port_return)
        
        portfolio_values.append(new_value)
        positions_history.append(weights)
        regimes_history.append(regime)
    
    # Compute metrics
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    
    cumulative = np.array(portfolio_values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return {
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values,
        'positions': positions_history,
        'regimes': regimes_history
    }

# Example usage
returns = np.random.normal(0.001, 0.01, size=(500, 1))
results = backtest(returns)

print(f"Final Value: ${results['final_value']:.2f}")
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
```

### Compare with Baseline

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

def compare_strategies(returns):
    """Compare HAF with buy-and-hold"""
    
    # HAF strategy
    haf = HausdorffAdaptiveFilter(n_assets=returns.shape[1])
    haf_values = [1.0]
    
    for ret in returns:
        regime, scale = haf.update(ret)
        weights = haf.get_action() * scale
        haf_return = np.dot(weights, ret)
        haf_values.append(haf_values[-1] * (1 + haf_return))
    
    # Buy and hold
    bh_values = [1.0]
    weights_bh = np.ones(returns.shape[1]) / returns.shape[1]
    
    for ret in returns:
        bh_return = np.dot(weights_bh, ret)
        bh_values.append(bh_values[-1] * (1 + bh_return))
    
    # Compare
    haf_sharpe = (np.mean(np.diff(haf_values)) / np.std(np.diff(haf_values))) * np.sqrt(252)
    bh_sharpe = (np.mean(np.diff(bh_values)) / np.std(np.diff(bh_values))) * np.sqrt(252)
    
    print("Strategy Comparison:")
    print(f"HAF:          Sharpe = {haf_sharpe:.3f}, Final = {haf_values[-1]:.3f}")
    print(f"Buy & Hold:   Sharpe = {bh_sharpe:.3f}, Final = {bh_values[-1]:.3f}")
    print(f"Improvement:  {(haf_sharpe - bh_sharpe) / bh_sharpe * 100:.1f}%")

# Test
returns = np.random.normal(0.001, 0.01, size=(300, 1))
compare_strategies(returns)
```

---

## Integration Patterns

### As a Risk Management Layer

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

class RiskManagedStrategy:
    """Wrap existing strategy with HAF risk management"""
    
    def __init__(self, base_strategy, n_assets=1):
        self.base_strategy = base_strategy
        self.haf = HausdorffAdaptiveFilter(n_assets=n_assets)
    
    def get_position(self, market_data):
        """Get risk-adjusted position"""
        # Base strategy signal
        base_signal = self.base_strategy(market_data)
        
        # HAF risk adjustment
        regime, risk_scale = self.haf.update(market_data['returns'])
        
        # Combine: use base signal but scale by HAF risk
        if regime == 'shift':
            # Emergency override
            return base_signal * 0.3
        else:
            return base_signal * risk_scale

# Example base strategy
def momentum_strategy(data):
    """Simple momentum: long if 10-day return positive"""
    return 1.0 if np.mean(data['returns'][-10:]) > 0 else 0.0

# Usage
strategy = RiskManagedStrategy(momentum_strategy)

# Simulate
for t in range(100):
    market_data = {
        'returns': np.random.normal(0.001, 0.01, size=min(t+1, 20))
    }
    position = strategy.get_position(market_data)
    print(f"t={t}: position={position:.2f}")
```

### Ensemble with Multiple Models

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

class EnsembleHAF:
    """Ensemble of HAF models with different sensitivities"""
    
    def __init__(self, n_assets=1):
        # Conservative, moderate, aggressive
        self.models = [
            HausdorffAdaptiveFilter(n_assets, rho_reset=1.5, safety_factor=0.5),
            HausdorffAdaptiveFilter(n_assets, rho_reset=1.2, safety_factor=0.3),
            HausdorffAdaptiveFilter(n_assets, rho_reset=1.1, safety_factor=0.2)
        ]
        self.weights = [0.3, 0.5, 0.2]  # Weight each model
    
    def update_and_act(self, observation):
        """Get ensemble action"""
        actions = []
        regimes = []
        
        for model in self.models:
            regime, scale = model.update(observation)
            action = model.get_action() * scale
            actions.append(action)
            regimes.append(regime)
        
        # Weighted average
        ensemble_action = sum(w * a for w, a in zip(self.weights, actions))
        
        return ensemble_action, regimes

# Usage
ensemble = EnsembleHAF(n_assets=1)

for t in range(100):
    ret = np.random.normal(0.001, 0.01, size=1)
    action, regimes = ensemble.update_and_act(ret)
    
    if t % 20 == 0:
        print(f"t={t}: action={action[0]:.3f}, regimes={regimes}")
```

### With Stop-Loss

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

class HAFWithStopLoss:
    """HAF with additional stop-loss protection"""
    
    def __init__(self, n_assets=1, stop_loss_pct=0.05):
        self.haf = HausdorffAdaptiveFilter(n_assets=n_assets)
        self.stop_loss_pct = stop_loss_pct
        self.cumulative_return = 0
        self.peak_return = 0
        self.stopped_out = False
    
    def update(self, observation):
        """Update with stop-loss check"""
        # Check stop-loss
        current_dd = self.peak_return - self.cumulative_return
        if current_dd > self.stop_loss_pct:
            self.stopped_out = True
            return 'stopped', 0.0
        
        # Normal HAF update
        regime, scale = self.haf.update(observation)
        
        # Track returns
        action = self.haf.get_action()
        period_return = np.dot(action * scale, observation)
        self.cumulative_return += period_return
        self.peak_return = max(self.peak_return, self.cumulative_return)
        
        return regime, scale

# Usage
haf_sl = HAFWithStopLoss(n_assets=1, stop_loss_pct=0.10)

for t in range(100):
    ret = np.random.normal(0.001, 0.01, size=1)
    regime, scale = haf_sl.update(ret)
    
    if regime == 'stopped':
        print(f"t={t}: STOPPED OUT at {haf_sl.cumulative_return*100:.2f}% return")
        break
```

---

## Tips & Best Practices

### 1. Parameter Selection
```python
# Start with defaults, then tune based on your data characteristics
# High volatility → increase window, increase rho_reset
# Low volatility → decrease window, decrease rho_reset
```

### 2. Warm-Up Period
```python
# Give HAF time to learn before making decisions
warmup = 50
for t, ret in enumerate(returns):
    regime, scale = haf.update(ret)
    if t < warmup:
        action = np.ones(n_assets) / n_assets  # Equal weight during warmup
    else:
        action = haf.get_action() * scale
```

### 3. Periodic Reset
```python
# Reset credal sets periodically to prevent drift
for t, ret in enumerate(returns):
    if t % 1000 == 0:
        haf.reset_credal_set()
    regime, scale = haf.update(ret)
```

### 4. Save/Load State
```python
import pickle

# Save HAF state
with open('haf_state.pkl', 'wb') as f:
    pickle.dump(haf, f)

# Load HAF state
with open('haf_state.pkl', 'rb') as f:
    haf = pickle.load(f)
```

---

For more examples, see `demo.py` and `examples/multi_asset_example.py`!
