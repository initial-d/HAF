"""
Statistical validation utilities
Implements: Bootstrap confidence intervals, p-values, performance metrics
"""

import numpy as np
from scipy import stats


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, confidence=0.95, block_size=10):
    """
    Compute bootstrap confidence interval with block bootstrap
    
    Args:
        data: Array of data points
        statistic_fn: Function to compute statistic (e.g., np.mean, compute_sharpe)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        block_size: Block size for block bootstrap (preserves serial dependence)
    
    Returns:
        (lower_bound, upper_bound, point_estimate)
    """
    n = len(data)
    n_blocks = n // block_size
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap: sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        bootstrap_sample = []
        
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n)
            bootstrap_sample.extend(data[start:end])
        
        bootstrap_sample = np.array(bootstrap_sample[:n])  # Trim to original length
        stat = statistic_fn(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    point = statistic_fn(data)
    
    return lower, upper, point


def compute_sharpe_ratio(returns, annualization_factor=252):
    """Compute annualized Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(annualization_factor)


def compute_max_drawdown(returns):
    """Compute maximum drawdown"""
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return np.min(drawdown)


def compute_calmar_ratio(returns, annualization_factor=252):
    """Compute Calmar ratio (annualized return / |max drawdown|)"""
    ann_return = np.mean(returns) * annualization_factor
    max_dd = compute_max_drawdown(returns)
    if max_dd == 0:
        return np.inf
    return ann_return / abs(max_dd)


def compute_turnover(positions):
    """Compute average turnover (position change)"""
    if len(positions) <= 1:
        return 0.0
    changes = np.abs(np.diff(positions))
    return np.mean(changes)


def bootstrap_comparison(returns_a, returns_b, metric_fn=compute_sharpe_ratio, 
                         n_bootstrap=1000, block_size=10):
    """
    Compare two strategies with bootstrap hypothesis test
    
    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
        metric_fn: Metric function to compare
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for block bootstrap
    
    Returns:
        p_value: Two-sided p-value for H0: metric_a == metric_b
        metric_a: Metric for strategy A
        metric_b: Metric for strategy B
    """
    metric_a = metric_fn(returns_a)
    metric_b = metric_fn(returns_b)
    observed_diff = metric_a - metric_b
    
    # Bootstrap null distribution (assuming no difference)
    n_a, n_b = len(returns_a), len(returns_b)
    combined = np.concatenate([returns_a, returns_b])
    n_total = len(combined)
    n_blocks = n_total // block_size
    
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap from combined data
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        bootstrap_sample = []
        
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n_total)
            bootstrap_sample.extend(combined[start:end])
        
        bootstrap_sample = np.array(bootstrap_sample[:n_total])
        
        # Split into two groups
        boot_a = bootstrap_sample[:n_a]
        boot_b = bootstrap_sample[n_a:n_a+n_b]
        
        diff = metric_fn(boot_a) - metric_fn(boot_b)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return p_value, metric_a, metric_b


def performance_metrics_with_ci(returns, positions, n_bootstrap=1000, confidence=0.95):
    """
    Compute performance metrics with bootstrap confidence intervals
    
    Args:
        returns: Array of returns
        positions: Array of positions
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Dictionary with metrics and CIs
    """
    results = {}
    
    # Sharpe ratio
    lower, upper, point = bootstrap_ci(returns, compute_sharpe_ratio, n_bootstrap, confidence)
    results['sharpe_ratio'] = point
    results['sharpe_ci'] = (lower, upper)
    
    # Max drawdown
    lower, upper, point = bootstrap_ci(returns, compute_max_drawdown, n_bootstrap, confidence)
    results['max_drawdown'] = point
    results['max_drawdown_ci'] = (lower, upper)
    
    # Calmar ratio
    lower, upper, point = bootstrap_ci(returns, compute_calmar_ratio, n_bootstrap, confidence)
    results['calmar_ratio'] = point
    results['calmar_ci'] = (lower, upper)
    
    # Final return
    results['final_return'] = np.sum(returns)
    
    # Turnover
    results['turnover'] = compute_turnover(positions)
    
    return results


def format_metric_with_ci(point, ci, is_percentage=False, decimals=2):
    """Format metric with confidence interval for display"""
    if is_percentage:
        return f"{point*100:.{decimals}f}% [{ci[0]*100:.{decimals}f}%, {ci[1]*100:.{decimals}f}%]"
    else:
        return f"{point:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def diebold_mariano_test(errors_a, errors_b):
    """
    Diebold-Mariano test for forecast comparison
    
    Args:
        errors_a: Forecast errors from model A
        errors_b: Forecast errors from model B
    
    Returns:
        p_value: p-value for H0: equal forecast accuracy
    """
    # Loss differential
    d = errors_a**2 - errors_b**2
    
    # Mean of differential
    d_bar = np.mean(d)
    
    # Standard error (with HAC correction)
    n = len(d)
    gamma_0 = np.var(d)
    
    # Simple approximation without HAC for brevity
    se = np.sqrt(gamma_0 / n)
    
    # Test statistic
    dm_stat = d_bar / se
    
    # p-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return p_value


def detect_regime_changes(regime_history, true_changes):
    """
    Evaluate regime detection performance
    
    Args:
        regime_history: List of detected regimes (0=stable, 1=uncertain, 2=shift)
        true_changes: List of true change point indices
    
    Returns:
        Dictionary with detection metrics
    """
    n = len(regime_history)
    detections = [i for i, r in enumerate(regime_history) if r == 2]  # Shift detections
    
    # For each true change, find nearest detection
    detection_lags = []
    for true_cp in true_changes:
        nearby_detections = [d for d in detections if abs(d - true_cp) <= 20]
        if nearby_detections:
            nearest = min(nearby_detections, key=lambda d: abs(d - true_cp))
            lag = nearest - true_cp
            detection_lags.append(lag)
        else:
            detection_lags.append(None)  # Missed
    
    # False positives (detections not near any true change)
    false_positives = 0
    for det in detections:
        is_fp = all(abs(det - tc) > 20 for tc in true_changes)
        if is_fp:
            false_positives += 1
    
    results = {
        'n_true_changes': len(true_changes),
        'n_detections': len(detections),
        'detection_lags': [lag for lag in detection_lags if lag is not None],
        'missed': detection_lags.count(None),
        'false_positives': false_positives,
        'detection_rate': 1 - detection_lags.count(None) / max(len(true_changes), 1)
    }
    
    return results
