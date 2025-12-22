# Quick Start Guide

## Installation (3 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/haf-implementation.git
cd haf-implementation

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Run Demo (1 minute)

```bash
# Run the basic demo with synthetic data
python demo.py
```

This will:
- Generate 700 periods of synthetic market data
- Simulate regime changes at t=300 (crisis) and t=400 (recovery)
- Run HAF algorithm and compare with baseline
- Save visualization to `haf_demo_results.png`

**Expected output:**
```
HAF Sharpe Ratio:    1.380
Baseline Sharpe:     0.820
HAF Improvement:     68.3%

First regime shift detected at t=301
Detection lag: 1 periods
```

## Run Tests (30 seconds)

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_haf.py::TestHAF::test_regime_detection_shift -v
```

## Try Multi-Asset Example (2 minutes)

```bash
# Run 3-asset portfolio example
python examples/multi_asset_example.py
```

This demonstrates HAF on a portfolio of correlated assets with different volatilities.

## Understanding the Output

### 1. Cumulative Returns
Shows HAF vs. baseline performance over time. HAF typically:
- Matches baseline in stable periods
- Significantly outperforms during regime shifts (red vertical lines)

### 2. Contraction Ratio (ρ)
The key regime detection signal:
- **ρ < 1**: System is contracting → stable regime
- **ρ ≈ 1**: Uncertain
- **ρ > 1.2**: System expanding → regime shift!

### 3. Credal Set Diameter
Measures epistemic uncertainty:
- **Low diameter**: Confident about regime → full position
- **High diameter**: Uncertain → reduced position

### 4. Position Sizes
Shows how HAF automatically scales exposure:
- Full exposure in stable regimes
- Reduced exposure when uncertainty spikes

## Key Concepts in 60 Seconds

**Credal Sets**: Instead of a single probability distribution, HAF maintains a SET of distributions (bull/bear/neutral). The "size" of this set measures how uncertain we are.

**Hausdorff Distance**: Measures how much the credal set changed between timesteps.

**Contraction Ratio**: Ratio of consecutive distance changes. In stable regimes, the credal set "contracts" (shrinks) toward the true distribution. During regime shifts, it expands.

**Pessimistic Decision**: HAF chooses actions that maximize worst-case expected return across all distributions in the credal set.

## Customization

### Adjust Sensitivity

```python
haf = HausdorffAdaptiveFilter(
    n_assets=1,
    rho_thresh=0.90,    # Lower = more cautious (detect regime sooner)
    rho_reset=1.30,     # Higher = less sensitive to shifts
    window=15,          # Larger = smoother detection
    safety_factor=0.5   # Higher = less aggressive reduction
)
```

### Use Your Own Data

```python
import numpy as np
from haf_core import HausdorffAdaptiveFilter

# Your market data (T x n_assets array)
returns = np.loadtxt('your_returns.csv')

haf = HausdorffAdaptiveFilter(n_assets=returns.shape[1])

for t, ret in enumerate(returns):
    regime, scale = haf.update(ret)
    weights = haf.get_action()
    final_position = weights * scale
    
    # Your trading logic here
    print(f"t={t}: regime={regime}, position={final_position}")
```

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'scipy'"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Tests fail with "AssertionError"
- **Solution**: This is likely due to random seed variations. Run tests multiple times.

**Issue**: No visualization generated
- **Solution**: Make sure matplotlib backend is configured. Try `export MPLBACKEND=Agg` before running.

**Issue**: HAF seems too conservative
- **Solution**: Increase `safety_factor` and/or increase `rho_reset` threshold

**Issue**: HAF not detecting regime changes
- **Solution**: Decrease `rho_reset` threshold and/or decrease `rho_thresh`

## Next Steps

1. **Read the paper**: See `fix_point.pdf` for theoretical foundations
2. **Experiment with parameters**: Try different thresholds on your data
3. **Extend to your use case**: Modify `get_action()` for your specific decision problem
4. **Compare with other methods**: Implement BOCPD, HMM, etc. from the paper

## Performance Expectations

Based on the paper's experiments:

| Scenario | HAF Advantage | When |
|----------|---------------|------|
| Crisis periods | 25-40% Sharpe improvement | High uncertainty |
| Stable regimes | ±5% (no significant diff) | Low uncertainty |
| Regime detection | 1-3 weeks earlier | vs. HMM/BOCPD |

**Key insight**: HAF doesn't always beat baselines - it excels specifically when epistemic uncertainty is high.

## Citation

If you use this code in research:

```bibtex
@article{
}
```

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Check README.md for detailed documentation
- **Contributing**: Pull requests welcome!

---

**Ready to start?** Run `python demo.py` now! 🚀
