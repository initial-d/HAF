"""
Data loading utilities for real financial data
Supports: yfinance (free), synthetic data generation
"""

import numpy as np
import warnings

# Try to import yfinance, provide fallback if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not installed. Install with: pip install yfinance")


def load_sp500(start_date='2010-01-01', end_date='2023-12-31'):
    """
    Load S&P 500 data
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        returns: Daily returns as numpy array
        dates: List of dates
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required. Install with: pip install yfinance")
    
    # Download S&P 500 ETF (SPY) as proxy
    ticker = yf.Ticker("SPY")
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("No data retrieved. Check dates and internet connection.")
    
    # Compute returns
    prices = data['Close'].values
    returns = np.diff(prices) / prices[:-1]
    dates = data.index[1:].tolist()
    
    return returns, dates


def load_bitcoin(start_date='2017-01-01', end_date='2023-12-31'):
    """
    Load Bitcoin data
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        returns: Hourly returns as numpy array
        dates: List of dates
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required. Install with: pip install yfinance")
    
    # Download Bitcoin (BTC-USD)
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(start=start_date, end=end_date, interval='1d')  # Daily for simplicity
    
    if data.empty:
        raise ValueError("No data retrieved. Check dates and internet connection.")
    
    prices = data['Close'].values
    returns = np.diff(prices) / prices[:-1]
    dates = data.index[1:].tolist()
    
    return returns, dates


def load_multi_asset(tickers, start_date='2015-01-01', end_date='2023-12-31'):
    """
    Load multiple assets
    
    Args:
        tickers: List of ticker symbols (e.g., ['SPY', 'TLT', 'GLD'])
        start_date: Start date
        end_date: End date
    
    Returns:
        returns: Array of shape (T, n_assets)
        dates: List of dates
        tickers: List of ticker names
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required. Install with: pip install yfinance")
    
    returns_list = []
    common_dates = None
    
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if data.empty:
                continue
            
            prices = data['Close'].values
            rets = np.diff(prices) / prices[:-1]
            dates_i = data.index[1:].tolist()
            
            if common_dates is None:
                common_dates = dates_i
            
            returns_list.append(rets)
        except Exception as e:
            print(f"Warning: Failed to load {ticker}: {e}")
    
    if not returns_list:
        raise ValueError("No data loaded for any ticker")
    
    # Align to shortest series
    min_len = min(len(r) for r in returns_list)
    returns_array = np.column_stack([r[:min_len] for r in returns_list])
    
    return returns_array, common_dates[:min_len], tickers


def generate_synthetic_regime_data(n_periods=700, regime_changes=None, seed=42):
    """
    Generate synthetic data with regime changes (same as demo.py but more flexible)
    
    Args:
        n_periods: Total number of periods
        regime_changes: List of change points (default: [300, 400])
        seed: Random seed
    
    Returns:
        returns: Array of returns
        true_regimes: Array indicating true regime (0=bull, 1=crisis, 2=recovery)
        change_points: List of change point indices
    """
    if regime_changes is None:
        regime_changes = [300, 400]
    
    np.random.seed(seed)
    returns = []
    true_regimes = []
    
    for t in range(n_periods):
        if t < regime_changes[0]:
            # Bull market
            ret = np.random.normal(0.0005, 0.015, size=1)
            regime = 0
        elif t < regime_changes[1]:
            # Crisis
            ret = np.random.normal(-0.0020, 0.03, size=1)
            regime = 1
        else:
            # Recovery
            ret = np.random.normal(0.0003, 0.02, size=1)
            regime = 2
        
        returns.append(ret)
        true_regimes.append(regime)
    
    return np.array(returns), np.array(true_regimes), regime_changes


def generate_high_snr_data(n_periods=500, sharpe_ratio=5.0, seed=42):
    """
    Generate high signal-to-noise ratio data (for negative test case)
    
    Args:
        n_periods: Number of periods
        sharpe_ratio: Target Sharpe ratio (annualized)
        seed: Random seed
    
    Returns:
        returns: Array of returns
    """
    np.random.seed(seed)
    
    # Daily Sharpe = Annual Sharpe / sqrt(252)
    daily_sharpe = sharpe_ratio / np.sqrt(252)
    
    # mean / std = daily_sharpe
    std = 0.003  # Low volatility
    mean = daily_sharpe * std
    
    returns = np.random.normal(mean, std, size=(n_periods, 1))
    return returns


def add_transaction_costs(returns, positions, cost_bps):
    """
    Add proportional transaction costs
    
    Args:
        returns: Asset returns
        positions: Position sizes
        cost_bps: Cost in basis points (e.g., 10 for 10bp = 0.1%)
    
    Returns:
        net_returns: Returns after transaction costs
    """
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * (cost_bps / 10000)  # Convert bp to decimal
    
    # Compute portfolio returns
    portfolio_returns = returns.flatten() * positions.flatten()
    
    # Subtract costs
    net_returns = portfolio_returns - costs
    
    return net_returns


# Preset configurations for common experiments
EXPERIMENT_CONFIGS = {
    'synthetic_regime': {
        'name': 'Synthetic Data with Regime Shifts',
        'data_fn': lambda: generate_synthetic_regime_data(700, [300, 400], 42),
        'description': 'Bull → Crisis → Recovery'
    },
    
    'sp500': {
        'name': 'S&P 500 (2010-2023)',
        'data_fn': lambda: (load_sp500('2010-01-01', '2023-12-31')[0].reshape(-1, 1), None, None),
        'description': 'Real US equity data'
    },
    
    'bitcoin': {
        'name': 'Bitcoin (2017-2023)',
        'data_fn': lambda: (load_bitcoin('2017-01-01', '2023-12-31')[0].reshape(-1, 1), None, None),
        'description': 'Real cryptocurrency data'
    },
    
    'high_snr': {
        'name': 'High SNR (Negative Test)',
        'data_fn': lambda: (generate_high_snr_data(500, 5.0, 42), None, None),
        'description': 'High signal-to-noise ratio environment'
    }
}


def load_experiment_data(experiment_name):
    """
    Load data for a specific experiment
    
    Args:
        experiment_name: Key from EXPERIMENT_CONFIGS
    
    Returns:
        returns: Return data
        true_regimes: True regime labels (if synthetic, else None)
        change_points: True change points (if synthetic, else None)
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    config = EXPERIMENT_CONFIGS[experiment_name]
    print(f"Loading: {config['name']}")
    print(f"Description: {config['description']}")
    
    return config['data_fn']()
