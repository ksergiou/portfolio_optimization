import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as dtr
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, efficient_frontier, DiscreteAllocation, get_latest_prices
import seaborn as sns
sns.set_style('darkgrid')

# Select the ETFs that comprise the portfolio
tickers = [
    # Equities
    'VUKE.L',  # FTSE100
    'VUSA.L',  # S&P500
    'VAPX.L',  # FTSE Developed Asia Pacific ex Japan
    'VJPN.L',  # FTSE Japan
    'VFEM.L',  # FTSE Emerging markets
    'VEUR.L',  # FTSE Developed Europe
    'VEVE.L',  # FTSE Developed World
    # Fixed Income
    'VETY.L',  # Eurozone Government Bond
    'VGOV.L',  # U.K. Gilt
    'VUTY.L',  # USD Treasury Bond
]

# Choose dates
start = dt.datetime(2017, 1, 1)
end = dt.datetime.today()

# Get a df with closing prices
df = pd.DataFrame()
for ticker in tickers:
    df[ticker[:-2]] = dtr.DataReader(ticker, 'yahoo', start, end)['Adj Close']

# Plot them and save png
def plot_stocks(tickers=tickers, df=df):
    plt.figure(figsize=(14,7))
    for ticker in df.columns.to_list():
        plt.plot(df[ticker], label=ticker)
    plt.title('Adj. Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Adj. Close Price (£)')
    plt.legend(loc='upper left')
    plt.savefig('ETF_prices_plot.png')
    #plt.show()
    plt.close()

plot_stocks()

# Create a df with daily returns
returns=df.pct_change()

# Annual covariance matrix - historically
cov_matrix=returns.cov() * 252

# Portfolio's historical returns and risk assuming equal weights
w=np.array( [1/len(tickers)]*len(tickers) )
portfolio_eq_var = w.T @ cov_matrix @ w
portfolio_eq_std=np.sqrt(portfolio_eq_var)
portfolio_eq_return=returns.mean().mean()*252
print('''The equally weighted portfolio had return {:0.4f} and volatility equal to {:0.4f}.'''.format(portfolio_eq_return, portfolio_eq_std) )

# VaR
por_rets=(returns*w).sum(axis=1)
var_hist = por_rets.quantile(0.05)
cvar_hist = por_rets [por_rets <= var_hist].mean()
print("VaR: {:.2f}%".format(100*var_hist))
print("CVaR: {:.2f}%".format(100*cvar_hist))
print('This value of the CVaR means that our average loss on the worst 5% of days will be {:.2f}%.\n'.format(cvar_hist*100))


# We find the expected return using the exponential moving average,
# which is a simple improvement over the mean historical return
exp_returns_est=expected_returns.ema_historical_return(df, span=800) # span was 500 by default

# Estimate the cov matrix using Ledoit-Wolf shrinkage
cov_matrix_est=risk_models.CovarianceShrinkage(df).ledoit_wolf(shrinkage_target='constant_variance')

# CVaR portfolio,
# long positions only
# maximise return subject to a CVaR constraint
CVaR=efficient_frontier.EfficientCVaR(exp_returns_est, returns.dropna(), beta=0.95, weight_bounds=(0, 1), solver=None, verbose=False, solver_options=None)
op_cvar=CVaR.efficient_risk(target_cvar=0.02)
w_cvar=CVaR.clean_weights()

# Get the discrete allocation
da=DiscreteAllocation(w_cvar, get_latest_prices(df), total_portfolio_value=1000)
allocation, leftover=da.lp_portfolio()
print('Allocation: ' ,allocation)
print('Leftover: ' ,leftover)
print( CVaR.portfolio_performance(verbose=True) );
portfolio_cvar_vol =np.array(list(w_cvar.values())).T @ cov_matrix_est @ np.array(list(w_cvar.values()))
print('Volatility:{:0.2f}% '.format(portfolio_cvar_vol*100 ))
# Find new expected annual return