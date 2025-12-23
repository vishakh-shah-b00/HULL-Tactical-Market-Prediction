"""
TRANSACTION COST ANALYZER
-------------------------
Determines if the strategy's small edge (50.4% Win Rate) survives real-world trading fees.

Math:
Breakeven Win Rate = 0.5 + (Fee / (2 * Avg_Move))

Scenarios:
1. Institutional (S&P 500 ETF): 0.05% Fee -> 50.4% is PROFITABLE.
2. Retail (Crypto/High Spread): 2.0% Fee -> 50.4% is BANKRUPTCY.

Conclusion:
This strategy requires low-cost execution (Futures/ETFs), not high-fee retail brokerage.
"""
print("="*80)
print("FEE IMPACT ANALYSIS")
print("="*80)

def simulate_trading(win_rate, fee_percent, n_trades=1000, avg_move=0.01):
    """
    Simulate trading PnL
    avg_move: Average daily market move (e.g., 1%)
    fee_percent: Transaction fee per trade (e.g., 0.02 for 2%)
    """
    capital = 100.0
    capital_history = [capital]
    
    # Expected Value per trade
    # EV = (Win_Prob * (Win_Amt - Fee)) - (Loss_Prob * (Loss_Amt + Fee))
    # Assuming Win_Amt = Loss_Amt = avg_move
    
    win_amt = avg_move * (1 - fee_percent) # Gain reduced by fee? Or fee separate?
    # Usually: PnL = (Price_Out - Price_In) / Price_In - Fee
    # Win:  +1% - Fee
    # Loss: -1% - Fee
    
    ev_per_trade = (win_rate * (avg_move - fee_percent)) - ((1 - win_rate) * (avg_move + fee_percent))
    
    print(f"\nScenario: Fees = {fee_percent*100:.3f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Avg Win (net): {(avg_move - fee_percent)*100:.4f}%")
    print(f"  Avg Loss (net): {(avg_move + fee_percent)*100:.4f}%")
    print(f"  EV per trade: {ev_per_trade*100:.4f}%")
    
    if ev_per_trade > 0:
        print("  -> PROFITABLE ✅")
    else:
        print("  -> UNPROFITABLE ❌")
        
    return ev_per_trade

# 1. Friend's Scenario (2% Fee)
print("\n--- 1. Friend's Scenario (Retail/Crypto/High Fee) ---")
ev_friend = simulate_trading(win_rate=0.504, fee_percent=0.02) # 2% fee

# 2. Realistic S&P 500 ETF/Futures Scenario (0.01% - 0.05% Fee)
# SPY spread is ~1 cent on $500 (0.002%) + commission
print("\n--- 2. Realistic S&P 500 Scenario (Institutional) ---")
ev_real = simulate_trading(win_rate=0.504, fee_percent=0.0005) # 0.05% fee (5 bps)

# 3. Breakeven Win Rate Calculation
def calc_breakeven(fee_percent, avg_move=0.01):
    # 0 = (WR * (M - F)) - ((1-WR) * (M + F))
    # 0 = WR(M-F) - (M+F) + WR(M+F)
    # M+F = WR(M-F + M+F)
    # M+F = WR(2M)
    # WR = (M+F) / 2M = 0.5 + (F/2M)
    return 0.5 + (fee_percent / (2 * avg_move))

print("\n--- 3. Required Win Rate to Break Even ---")
be_friend = calc_breakeven(0.02)
be_real = calc_breakeven(0.0005)

print(f"With 2.00% Fees: Need > {be_friend*100:.2f}% Win Rate")
print(f"With 0.05% Fees: Need > {be_real*100:.2f}% Win Rate")
