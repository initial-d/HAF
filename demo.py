"""
Demo script showing HAF performance on synthetic market data with regime shifts
"""

import numpy as np
import matplotlib.pyplot as plt
from haf_core import HausdorffAdaptiveFilter


def generate_synthetic_data(n_periods: int = 700, 
                            regime_changes: list = [300, 400],
                            seed: int = 42) -> np.ndarray:
    """
    Generate synthetic return data with regime changes
    
    Regimes:
    - [0, 300): Bull market (0.05% daily return, 1% volatility)
    - [300, 400): Crisis (−0.15% daily return, 3% volatility)
    - [400, 700): Recovery (0.03% daily return, 1.5% volatility)
    """
    np.random.seed(seed)
    returns = []
    
    for t in range(n_periods):
        if t < regime_changes[0]:
            # Bull market
            r = np.random.normal(0.0005, 0.01, size=1)
        elif t < regime_changes[1]:
            # Crisis
            r = np.random.normal(-0.0015, 0.03, size=1)
        else:
            # Recovery
            r = np.random.normal(0.0003, 0.015, size=1)
        
        returns.append(r)
    
    return np.array(returns)


def compute_performance_metrics(returns: np.ndarray, 
                                positions: np.ndarray) -> dict:
    """Compute Sharpe ratio, max drawdown, etc."""
    # Portfolio returns
    portfolio_returns = returns.flatten() * positions.flatten()
    
    # Cumulative returns
    cumulative_returns = np.cumsum(portfolio_returns)
    
    # Sharpe ratio (annualized, assuming daily data)
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.min(drawdown)
    
    # Turnover
    position_changes = np.abs(np.diff(positions.flatten()))
    turnover = np.mean(position_changes)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_return': cumulative_returns[-1],
        'turnover': turnover,
        'cumulative_returns': cumulative_returns
    }


class SimpleBaseline:
    """Simple rolling mean baseline for comparison"""
    
    def __init__(self, window: int = 60):
        self.window = window
        self.history = []
    
    def update(self, observation: np.ndarray):
        self.history.append(observation[0])
        if len(self.history) > self.window:
            self.history.pop(0)
    
    def get_action(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.array([0.5])
        
        # Simple: positive mean -> long, negative mean -> short
        mean_return = np.mean(self.history)
        if mean_return > 0:
            return np.array([1.0])
        else:
            return np.array([0.0])


def run_demo():
    """Run complete demo comparing HAF vs baseline"""
    print("=" * 60)
    print("Hausdorff-Adaptive Filter (HAF) Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic market data...")
    returns = generate_synthetic_data()
    print(f"   Generated {len(returns)} periods with regime changes at t=300, t=400")
    
    # Initialize methods
    print("\n2. Initializing algorithms...")
    haf = HausdorffAdaptiveFilter(n_assets=1)
    baseline = SimpleBaseline(window=60)
    
    # Run online learning
    print("\n3. Running online learning...")
    haf_positions = []
    baseline_positions = []
    haf_regimes = []
    haf_diameters = []
    haf_rhos = []
    
    for t, ret in enumerate(returns):
        # HAF update
        regime, position_scale = haf.update(ret)
        action = haf.get_action()
        haf_positions.append(action[0] * position_scale)
        
        metrics = haf.get_metrics()
        haf_regimes.append(metrics['regime'])
        haf_diameters.append(metrics['diameter'])
        if metrics['rho'] != 0:
            haf_rhos.append(metrics['rho'])
        
        # Baseline update
        baseline.update(ret)
        baseline_positions.append(baseline.get_action()[0])
        
        if (t + 1) % 100 == 0:
            print(f"   Progress: {t+1}/{len(returns)}")
    
    haf_positions = np.array(haf_positions)
    baseline_positions = np.array(baseline_positions)
    
    # Compute performance
    print("\n4. Computing performance metrics...")
    haf_perf = compute_performance_metrics(returns, haf_positions)
    baseline_perf = compute_performance_metrics(returns, baseline_positions)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nHAF Performance:")
    print(f"  Sharpe Ratio:    {haf_perf['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:    {haf_perf['max_drawdown']*100:.2f}%")
    print(f"  Final Return:    {haf_perf['final_return']*100:.2f}%")
    print(f"  Turnover:        {haf_perf['turnover']:.3f}")
    
    print("\nBaseline Performance:")
    print(f"  Sharpe Ratio:    {baseline_perf['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:    {baseline_perf['max_drawdown']*100:.2f}%")
    print(f"  Final Return:    {baseline_perf['final_return']*100:.2f}%")
    print(f"  Turnover:        {baseline_perf['turnover']:.3f}")
    
    improvement = ((haf_perf['sharpe_ratio'] - baseline_perf['sharpe_ratio']) 
                   / baseline_perf['sharpe_ratio'] * 100)
    print(f"\nHAF Improvement: {improvement:.1f}%")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative returns
    axes[0].plot(haf_perf['cumulative_returns'], label='HAF', linewidth=2)
    axes[0].plot(baseline_perf['cumulative_returns'], label='Baseline', 
                 linewidth=2, alpha=0.7)
    axes[0].axvline(300, color='red', linestyle='--', alpha=0.5, label='Regime Change')
    axes[0].axvline(400, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('Cumulative Returns Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Contraction ratio (regime detection signal)
    if len(haf_rhos) > 0:
        axes[1].plot(haf_rhos, linewidth=1.5, color='blue')
        axes[1].axhline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[1].axhline(haf.rho_reset, color='red', linestyle='--', 
                       label=f'Reset threshold ({haf.rho_reset})')
        axes[1].axvline(300, color='red', linestyle='--', alpha=0.5)
        axes[1].axvline(400, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Contraction Ratio ρ')
        axes[1].set_title('HAF Contraction Ratio (Regime Detection Signal)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Credal set diameter (epistemic uncertainty)
    axes[2].plot(haf_diameters, linewidth=1.5, color='purple')
    axes[2].axvline(300, color='red', linestyle='--', alpha=0.5)
    axes[2].axvline(400, color='red', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Diameter')
    axes[2].set_title('Credal Set Diameter (Epistemic Uncertainty)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Position sizes
    axes[3].plot(haf_positions, label='HAF Position', linewidth=1.5, alpha=0.8)
    axes[3].plot(baseline_positions, label='Baseline Position', 
                linewidth=1.5, alpha=0.6)
    axes[3].axvline(300, color='red', linestyle='--', alpha=0.5)
    axes[3].axvline(400, color='red', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Position Size')
    axes[3].set_xlabel('Time Period')
    axes[3].set_title('Position Sizes Over Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('haf_demo_results.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'haf_demo_results.png'")
    
    # Detection timing analysis
    print("\n6. Regime Detection Analysis:")
    regime_changes_detected = []
    for i, regime in enumerate(haf_regimes):
        if regime == 2:  # Shift detected
            regime_changes_detected.append(i)
    
    if len(regime_changes_detected) > 0:
        first_detection = regime_changes_detected[0]
        print(f"   First regime shift detected at t={first_detection}")
        print(f"   Actual regime change at t=300")
        print(f"   Detection lag: {first_detection - 300} periods")
        
        if len(regime_changes_detected) > 1:
            second_detection = [d for d in regime_changes_detected if d > 350]
            if second_detection:
                print(f"   Second regime shift detected at t={second_detection[0]}")
                print(f"   Actual regime change at t=400")
                print(f"   Detection lag: {second_detection[0] - 400} periods")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_demo()
