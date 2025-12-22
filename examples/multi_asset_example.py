"""
Multi-asset portfolio example using HAF

This example demonstrates HAF on a portfolio of multiple correlated assets.
"""

import numpy as np
import matplotlib.pyplot as plt
from haf_core import HausdorffAdaptiveFilter


def generate_correlated_returns(n_periods: int = 500,
                                n_assets: int = 3,
                                regime_changes: list = [200, 350],
                                seed: int = 42) -> np.ndarray:
    """
    Generate synthetic multi-asset returns with regime changes
    
    Returns shape: (n_periods, n_assets)
    """
    np.random.seed(seed)
    
    # Correlation matrix
    corr = np.array([[1.0, 0.5, 0.3],
                     [0.5, 1.0, 0.4],
                     [0.3, 0.4, 1.0]])
    
    returns = []
    
    for t in range(n_periods):
        if t < regime_changes[0]:
            # Bull market
            mean = np.array([0.0008, 0.0006, 0.0007])
            vol = np.array([0.01, 0.012, 0.011])
        elif t < regime_changes[1]:
            # Crisis
            mean = np.array([-0.002, -0.0015, -0.0018])
            vol = np.array([0.03, 0.028, 0.032])
        else:
            # Recovery
            mean = np.array([0.0004, 0.0005, 0.0003])
            vol = np.array([0.015, 0.014, 0.016])
        
        # Generate correlated returns
        cov = np.outer(vol, vol) * corr
        ret = np.random.multivariate_normal(mean, cov)
        returns.append(ret)
    
    return np.array(returns)


def run_multi_asset_example():
    """Run multi-asset portfolio example"""
    print("=" * 70)
    print("Multi-Asset Portfolio Example with HAF")
    print("=" * 70)
    
    # Parameters
    n_assets = 3
    n_periods = 500
    
    # Generate data
    print(f"\n1. Generating {n_periods} periods of {n_assets}-asset returns...")
    returns = generate_correlated_returns(n_periods, n_assets)
    print(f"   Data shape: {returns.shape}")
    print(f"   Regime changes at t=200 (crisis) and t=350 (recovery)")
    
    # Initialize HAF
    print("\n2. Initializing HAF for multi-asset portfolio...")
    haf = HausdorffAdaptiveFilter(
        n_assets=n_assets,
        rho_thresh=0.95,
        rho_reset=1.2,
        window=10
    )
    
    # Run online learning
    print("\n3. Running online portfolio optimization...")
    portfolio_returns = []
    portfolio_weights_history = []
    position_scales = []
    uncertainty_history = []
    
    for t, ret in enumerate(returns):
        # HAF update
        regime, position_scale = haf.update(ret)
        
        # Get portfolio weights
        weights = haf.get_action()
        
        # Scale by position scale
        scaled_weights = weights * position_scale
        
        # Compute portfolio return
        portfolio_ret = np.dot(scaled_weights, ret)
        
        portfolio_returns.append(portfolio_ret)
        portfolio_weights_history.append(scaled_weights)
        position_scales.append(position_scale)
        
        metrics = haf.get_metrics()
        uncertainty_history.append(metrics['diameter'])
        
        if (t + 1) % 100 == 0:
            print(f"   Progress: {t+1}/{n_periods} - Regime: {regime}, Scale: {position_scale:.2f}")
    
    portfolio_returns = np.array(portfolio_returns)
    portfolio_weights_history = np.array(portfolio_weights_history)
    
    # Compute performance
    print("\n4. Computing performance metrics...")
    cumulative_returns = np.cumsum(portfolio_returns)
    sharpe_ratio = (np.mean(portfolio_returns) / np.std(portfolio_returns)) * np.sqrt(252)
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.min(drawdown)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nPortfolio Performance:")
    print(f"  Sharpe Ratio:     {sharpe_ratio:.3f}")
    print(f"  Final Return:     {cumulative_returns[-1]*100:.2f}%")
    print(f"  Max Drawdown:     {max_drawdown*100:.2f}%")
    print(f"  Avg Uncertainty:  {np.mean(uncertainty_history):.4f}")
    
    # Analyze regime periods
    print(f"\nRegime Analysis:")
    bull_returns = portfolio_returns[:200]
    crisis_returns = portfolio_returns[200:350]
    recovery_returns = portfolio_returns[350:]
    
    print(f"  Bull Market SR:    {(np.mean(bull_returns)/np.std(bull_returns))*np.sqrt(252):.3f}")
    print(f"  Crisis SR:         {(np.mean(crisis_returns)/np.std(crisis_returns))*np.sqrt(252):.3f}")
    print(f"  Recovery SR:       {(np.mean(recovery_returns)/np.std(recovery_returns))*np.sqrt(252):.3f}")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    
    # Plot 1: Cumulative returns
    axes[0].plot(cumulative_returns, linewidth=2, color='darkblue')
    axes[0].axvline(200, color='red', linestyle='--', alpha=0.5, label='Regime Changes')
    axes[0].axvline(350, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('Multi-Asset Portfolio Cumulative Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Portfolio weights over time
    for i in range(n_assets):
        axes[1].plot(portfolio_weights_history[:, i], 
                    label=f'Asset {i+1}', linewidth=1.5, alpha=0.7)
    axes[1].axvline(200, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(350, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Portfolio Weights Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Position scale (total exposure)
    axes[2].plot(position_scales, linewidth=1.5, color='green')
    axes[2].axvline(200, color='red', linestyle='--', alpha=0.5)
    axes[2].axvline(350, color='red', linestyle='--', alpha=0.5)
    axes[2].axhline(1.0, color='black', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Position Scale')
    axes[2].set_title('Total Portfolio Exposure (Position Scaling)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Epistemic uncertainty
    axes[3].plot(uncertainty_history, linewidth=1.5, color='purple')
    axes[3].axvline(200, color='red', linestyle='--', alpha=0.5)
    axes[3].axvline(350, color='red', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Credal Set Diameter')
    axes[3].set_xlabel('Time Period')
    axes[3].set_title('Epistemic Uncertainty Over Time')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_asset_results.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'multi_asset_results.png'")
    
    # Weight allocation analysis
    print("\n6. Portfolio Weight Statistics:")
    print("\n   Bull Market (t=0-200):")
    bull_weights = portfolio_weights_history[:200]
    for i in range(n_assets):
        print(f"     Asset {i+1}: {np.mean(bull_weights[:, i]):.3f} ± {np.std(bull_weights[:, i]):.3f}")
    
    print("\n   Crisis (t=200-350):")
    crisis_weights = portfolio_weights_history[200:350]
    for i in range(n_assets):
        print(f"     Asset {i+1}: {np.mean(crisis_weights[:, i]):.3f} ± {np.std(crisis_weights[:, i]):.3f}")
    
    print("\n   Recovery (t=350-500):")
    recovery_weights = portfolio_weights_history[350:]
    for i in range(n_assets):
        print(f"     Asset {i+1}: {np.mean(recovery_weights[:, i]):.3f} ± {np.std(recovery_weights[:, i]):.3f}")
    
    print("\n" + "=" * 70)
    print("Multi-asset example completed!")
    print("=" * 70)


if __name__ == '__main__':
    run_multi_asset_example()
