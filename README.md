# Hausdorff-Adaptive Filter (HAF)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the **Hausdorff-Adaptive Filter (HAF)** algorithm from the paper:

> **"Robust Online Learning in Non-Stationary Markets: A Credal Set Approach with Uncertainty Quantification"**  
> Yimin Du, December 2025

## Overview

HAF is an online learning algorithm designed for regime detection and adaptive decision-making in non-stationary environments (especially financial markets). It uses **credal sets** (convex sets of probability distributions) to quantify epistemic uncertainty and detect regime changes through contraction ratio monitoring.

### Key Features

- **Epistemic Uncertainty Quantification**: Uses credal sets to distinguish between uncertainty due to lack of data vs. inherent randomness
- **Automatic Regime Detection**: Monitors Hausdorff contraction ratios to detect regime shifts without manual tuning
- **Principled Risk Management**: Automatically scales positions based on credal set diameter (epistemic uncertainty)
- **Theoretical Guarantees**: Geometric convergence in stable regimes, probabilistic regime detection bounds

### Core Idea

In stable market regimes, credal sets contract geometrically toward a fixed point. When a regime shift occurs, this contractivity breaks down - the credal set expands rather than contracts, providing an early warning signal.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/haf-implementation.git
cd haf-implementation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from haf_core import HausdorffAdaptiveFilter
import numpy as np

# Initialize HAF
haf = HausdorffAdaptiveFilter(
    n_assets=1,
    rho_thresh=0.95,    # Stable regime threshold
    rho_reset=1.2,      # Regime shift threshold
    window=10           # Moving average window
)

# Online learning loop
for observation in market_data:
    # Update with new observation
    regime, position_scale = haf.update(observation)
    
    # Get recommended action (portfolio weights)
    action = haf.get_action()
    
    # Scale position based on uncertainty
    final_position = action * position_scale
    
    # Get monitoring metrics
    metrics = haf.get_metrics()
    print(f"Regime: {regime}, Uncertainty: {metrics['diameter']:.4f}")
```

## Running the Demo

The demo generates synthetic market data with known regime changes and compares HAF against a simple baseline:

```bash
python demo.py
```

This will:
1. Generate 700 periods of synthetic data with regime changes at t=300 (crisis) and t=400 (recovery)
2. Run HAF and baseline algorithms
3. Compute performance metrics (Sharpe ratio, max drawdown, etc.)
4. Create visualization showing:
   - Cumulative returns comparison
   - Contraction ratio (regime detection signal)
   - Credal set diameter (epistemic uncertainty)
   - Position sizes over time

### Expected Results

From the paper's experiments, HAF achieves:
- **25-40% higher Sharpe ratios** during turbulent periods vs. baselines
- **Earlier regime detection** (e.g., 3 weeks before peak in 2015 China crash)
- **Comparable performance** in stable regimes (no over-conservatism)

## Project Structure

```
haf-implementation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── haf_core.py              # Core HAF implementation
├── demo.py                  # Synthetic data demo
├── tests/
│   └── test_haf.py          # Unit tests
└── examples/
    └── multi_asset_example.py  # Multi-asset portfolio example
```

## Core Components

### 1. Credal Sets (`CredalSet` class)

Represents a convex set of probability distributions through K extreme distributions (typically K=3):
- **Bull regime**: Positive mean, low volatility
- **Bear regime**: Negative mean, high volatility  
- **Neutral regime**: Zero mean, medium volatility

```python
credal_set = CredalSet([bull_dist, bear_dist, neutral_dist])
diameter = credal_set.diameter()  # Epistemic uncertainty measure
```

### 2. Hausdorff Distance

The Hausdorff metric measures the "distance" between two credal sets:

```python
d_H = credal_set_t.hausdorff_distance(credal_set_{t-1})
```

### 3. Contraction Ratio (ρ)

The key signal for regime detection:

```
ρ_t = d_H(P_t, P_{t-1}) / d_H(P_{t-1}, P_{t-2})
```

- **ρ < 1**: Contracting (stable regime)
- **ρ ≈ 1**: Uncertain
- **ρ > 1.2**: Expanding (regime shift detected!)

### 4. Pessimistic Decision Rule

Choose actions that maximize worst-case expected utility across all distributions in the credal set:

```python
action = argmax_a min_{P ∈ credal_set} E_P[utility(a)]
```

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho_thresh` | 0.95 | Threshold for identifying stable regimes |
| `rho_reset` | 1.2 | Threshold for detecting regime shifts |
| `window` | 10 | Moving average window for ρ smoothing |
| `safety_factor` | 0.3 | Position scaling during detected regime shifts |

## Theoretical Guarantees

### Theorem 4.1: Convergence in Stable Regimes

In stable regimes, credal sets converge geometrically:

```
E[d_H(P_t, P*)] ≤ d_H(P_0, P*) · τ^t
```

where τ < 1 is the contraction rate.

### Theorem 4.2: Regime Detection

When a regime shift occurs with separation Δ:

```
P(ρ_t ≥ 1 + δ) ≥ 1 - exp(-c·n·Δ²/σ²)
```

Larger regime changes are detected with higher probability.

### Theorem 4.3: Regret Bounds

Cumulative regret is controlled by epistemic uncertainty:

```
R_T ≤ U_max · d_H(P_0, {P*}) · (1-τ^T)/(1-τ)
```

## When to Use HAF

**Recommended for:**
- Non-stationary environments with regime shifts
- Need for interpretable uncertainty quantification
- Moderate transaction costs (< 10 basis points)
- Daily or lower frequency rebalancing

**May not help when:**
- Extended stable regimes with high signal-to-noise
- Very high transaction costs (> 20 bps)
- High-frequency trading requirements

## Extensions & Future Work

The paper suggests several extensions:
1. **Decompose epistemic vs. aleatoric uncertainty** formally
2. **Incorporate transaction costs** into theoretical guarantees
3. **Scale to high dimensions** via factor models
4. **Applications beyond finance**: robotics, healthcare, recommendation systems

## Citation

If you use this implementation, please cite:

```bibtex
@article{,
 
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on credal set theory by Walley (1991) and fixed-point theorems by Caprio et al. (2025)

## Contact

For questions or issues, please open an GitHub issue or contact the maintainer.

---

**Note**: This is a research implementation for educational purposes. Not intended for production trading without proper risk management and validation.
