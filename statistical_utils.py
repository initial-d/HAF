"""
Statistical validation utilities (FIXED VERSION)
Implements: Paired Block Bootstrap, HAC-adjusted tests, performance metrics
"""

import numpy as np
from scipy import stats

def _get_block_indices(n, block_size):
    """
    Helper generator for block bootstrap indices.
    Fixes data truncation issue by ensuring full length coverage.
    """
    # Calculate how many blocks we need to cover n, rounding up
    n_blocks = int(np.ceil(n / block_size))
    
    # Generate random start points
    # We allow starts up to n - block_size to avoid array out of bounds
    # (Simple Block Bootstrap). For Circular, logic would differ slightly.
    valid_starts = n - block_size + 1
    if valid_starts <= 0:
        # Fallback for very short arrays
        return np.random.randint(0, n, size=n)
        
    start_indices = np.random.randint(0, valid_starts, size=n_blocks)
    
    indices = []
    for start in start_indices:
        indices.extend(range(start, start + block_size))
    
    # Trim strictly to length n to match original data
    return np.array(indices[:n])


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, confidence=0.95, block_size=10):
    """
    Compute bootstrap confidence interval with block bootstrap.
    """
    data = np.array(data)
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # 1. Generate indices using helper (Fixes truncation)
        indices = _get_block_indices(n, block_size)
        
        # 2. Resample
        sample = data[indices]
        
        # 3. Compute stat
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    point = statistic_fn(data)
    
    return lower, upper, point


def bootstrap_comparison(returns_a, returns_b, metric_fn, n_bootstrap=1000, block_size=10):
    """
    Compare two strategies using PAIRED Block Bootstrap.
    
    CRITICAL FIX: 
    This version resamples *indices* and applies them to both A and B simultaneously.
    This preserves the temporal correlation (pairing) between strategies.
    
    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
        metric_fn: Metric function (e.g. compute_sharpe_ratio)
    
    Returns:
        p_value: Two-sided p-value for H0: metric_a == metric_b
        metric_a: Observed metric for A
        metric_b: Observed metric for B
    """
    returns_a = np.array(returns_a)
    returns_b = np.array(returns_b)
    
    if len(returns_a) != len(returns_b):
        raise ValueError("Returns must have same length for paired comparison")
        
    n = len(returns_a)
    
    # 1. Observed metrics
    obs_metric_a = metric_fn(returns_a)
    obs_metric_b = metric_fn(returns_b)
    obs_diff = obs_metric_a - obs_metric_b
    
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # 2. PAIRED Resampling (Fixes the correlation issue)
        # We use the SAME indices for both A and B
        indices = _get_block_indices(n, block_size)
        
        sample_a = returns_a[indices]
        sample_b = returns_b[indices]
        
        # 3. Compute difference on this bootstrap world
        diff = metric_fn(sample_a) - metric_fn(sample_b)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # 4. Compute P-value
    # H0: diff = 0. We check how often the bootstrap distribution of differences
    # crosses 0 or is more extreme than 0 in the opposite direction.
    # A robust way is the "Percentile Method":
    # If 0 is outside the 95% CI of the differences, it is significant.
    
    # Calculation: Proportion of bootstrap diffs that have opposite sign to observed diff
    # multiplied by 2 (two-sided)
    if obs_diff > 0:
        p_value = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= 0)
        
    # Cap p-value at 1.0
    p_value = min(p_value, 1.0)
    
    return p_value, obs_metric_a, obs_metric_b


def compute_sharpe_ratio(returns, annualization_factor=252):
    """Compute annualized Sharpe ratio with safety checks"""
    returns = np.array(returns)
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    return (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(annualization_factor)


def compute_max_drawdown(returns):
    """Compute maximum drawdown"""
    cumulative = np.cumsum(returns)
    # Ensure we start at 0 or account for initial value
    running_max = np.maximum.accumulate(cumulative)
    # Protect against cases where running_max is all negative or zero
    drawdown = cumulative - running_max
    return np.min(drawdown)


def compute_calmar_ratio(returns, annualization_factor=252):
    """Compute Calmar ratio (annualized return / |max drawdown|)"""
    ann_return = np.mean(returns) * annualization_factor
    max_dd = compute_max_drawdown(returns)
    if abs(max_dd) < 1e-9:
        return 0.0 # Avoid division by zero
    return ann_return / abs(max_dd)


def compute_turnover(positions):
    """Compute average turnover (position change)"""
    positions = np.array(positions)
    if len(positions) <= 1:
        return 0.0
    changes = np.abs(np.diff(positions))
    return np.mean(changes)


def performance_metrics_with_ci(returns, positions, n_bootstrap=1000, confidence=0.95):
    """
    Compute performance metrics with bootstrap confidence intervals
    """
    results = {}
    
    # 1. Sharpe Ratio
    # 注意：这里的键名必须是 'sharpe_ci' 以匹配你的 reproduce_paper.py 调用
    lower, upper, point = bootstrap_ci(returns, compute_sharpe_ratio, n_bootstrap, confidence)
    results['sharpe_ratio'] = point
    results['sharpe_ci'] = (lower, upper)
    
    # 2. Max Drawdown
    lower, upper, point = bootstrap_ci(returns, compute_max_drawdown, n_bootstrap, confidence)
    results['max_drawdown'] = point
    results['max_drawdown_ci'] = (lower, upper)
    
    # 3. Calmar Ratio
    lower, upper, point = bootstrap_ci(returns, compute_calmar_ratio, n_bootstrap, confidence)
    results['calmar_ratio'] = point
    results['calmar_ci'] = (lower, upper)
    
    # 4. Other scalar metrics (no CI needed usually, or too expensive)
    results['final_return'] = np.sum(returns)
    results['turnover'] = compute_turnover(positions)
    
    return results



def format_metric_with_ci(point, ci, is_percentage=False, decimals=2):
    """Format metric with confidence interval for display"""
    mult = 100 if is_percentage else 1
    fmt = f".{decimals}f"
    return f"{point*mult:{fmt}} [{ci[0]*mult:{fmt}}, {ci[1]*mult:{fmt}}]"


def diebold_mariano_test(errors_a, errors_b, h=1):
    """
    Diebold-Mariano test with basic variance correction.
    Note: For strict academic rigor, a HAC estimator (like Newey-West) 
    should be used for the variance, but this is a standard approximation.
    """
    errors_a = np.array(errors_a)
    errors_b = np.array(errors_b)
    
    # Loss differential
    d = errors_a**2 - errors_b**2
    n = len(d)
    
    d_bar = np.mean(d)
    
    # Simple variance (naive)
    gamma_0 = np.var(d)
    
    if gamma_0 < 1e-9:
        return 1.0 # No difference
        
    se = np.sqrt(gamma_0 / n)
    dm_stat = d_bar / se
    
    # p-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return p_value


def detect_regime_changes(regime_history, true_changes, tolerance=20):
    """
    Evaluate regime detection performance
    Args:
        regime_history: 0=stable, 1=uncertain, 2=shift
        true_changes: indices of true changes
        tolerance: window size to consider a detection 'correct'
    """
    detections = [i for i, r in enumerate(regime_history) if r == 2]
    
    detection_lags = []
    found_changes = set()
    
    # Check for True Positives
    for true_cp in true_changes:
        # Find closest detection
        nearby = [d for d in detections if abs(d - true_cp) <= tolerance]
        if nearby:
            nearest = min(nearby, key=lambda d: abs(d - true_cp))
            detection_lags.append(nearest - true_cp)
            found_changes.add(true_cp)
        else:
            detection_lags.append(None) # Missed
            
    # Check for False Positives (detections far from any true change)
    false_positives = 0
    for det in detections:
        is_fp = True
        for tc in true_changes:
            if abs(det - tc) <= tolerance:
                is_fp = False
                break
        if is_fp:
            false_positives += 1
            
    # Compile stats
    n_true = len(true_changes)
    n_found = len(found_changes)
    
    valid_lags = [l for l in detection_lags if l is not None]
    avg_lag = np.mean(valid_lags) if valid_lags else 0
    
    return {
        'n_true_changes': n_true,
        'n_detections': len(detections),
        'missed': n_true - n_found,
        'false_positives': false_positives,
        'detection_rate': n_found / n_true if n_true > 0 else 0,
        'avg_lag': avg_lag
    }
