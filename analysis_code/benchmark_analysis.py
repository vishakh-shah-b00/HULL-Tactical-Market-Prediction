"""
BENCHMARK PERFORMANCE ANALYZER
------------------------------
Calculates the "Buy and Hold" performance of the S&P 500 for comparison.

Why?
- A strategy Sharpe of 0.35 is meaningless without context.
- If the S&P 500 Sharpe was 1.0, we failed.
- If the S&P 500 Sharpe was 0.2, we outperformed significantly.

Output:
- S&P 500 Annualized Sharpe Ratio over the 36-year history.
"""
print("="*80)
print("BENCHMARK ANALYSIS: S&P 500 vs MODEL")
print("="*80)

# Load data
train = pd.read_csv('train.csv')

# The target 'market_forward_excess_returns' IS the market's excess return
# So calculating Sharpe on this column gives us the "Buy & Hold" Sharpe for excess returns
market_returns = train['market_forward_excess_returns']

# Calculate Benchmark Sharpe (Buy & Hold)
mean_ret = market_returns.mean()
std_ret = market_returns.std()
sharpe = (mean_ret / std_ret) * np.sqrt(252)

print(f"\nDataset Range: {len(train)} days (~{len(train)/252:.1f} years)")
print(f"S&P 500 (Excess Return) Stats:")
print(f"  Mean Daily Return: {mean_ret:.6f}")
print(f"  Daily Volatility:  {std_ret:.6f}")
print(f"  Annualized Sharpe: {sharpe:.4f}")

print("\nComparison:")
print(f"  S&P 500 Buy & Hold Sharpe: {sharpe:.4f}")
print(f"  Our Model CV Sharpe:       0.3544")
print(f"  Our Model Holdout Sharpe:  2.8936")

print("\nInterpretation:")
if sharpe < 0.35:
    print("  -> We are BEATING the market average (Alpha > Market).")
else:
    print("  -> We are lagging the market average.")
    
# Check distribution of market returns
print(f"\nMarket Return Distribution:")
print(f"  Min: {market_returns.min():.4f}")
print(f"  Max: {market_returns.max():.4f}")
print(f"  Skew: {market_returns.skew():.4f}")
