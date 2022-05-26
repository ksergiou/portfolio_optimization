import pandas as pd
import sys
import numpy as np
import datetime as dt
import pandas_datareader.data as dtr
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier, DiscreteAllocation, get_latest_prices, \
    EfficientCVaR, objective_functions, plotting
import seaborn as sns

sns.set_style('darkgrid')

# Print everything in this file
sys.stdout = open('output.txt', 'w')  # Change the standard output to the file we created.
#sys.stdout = open('README.md', 'w')

# Set the amount of £ available
pf_value = 1000

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
def plot_save_stocks(tickers=tickers, df=df, name='ETF_prices_plot.png'):
    plt.figure(figsize=(14, 7))
    for ticker in df.columns.to_list():
        plt.plot(df[ticker], label=ticker)
    plt.title('Adj. Close Prices (£)')
    plt.xlabel('Date')
    #plt.ylabel('Adj. Close Price (£)')
    plt.legend(loc='upper left')
    plt.savefig(name)
    # plt.show()
    plt.close()


plot_save_stocks()

# Create a df with daily returns
returns = df.pct_change()

# Annual covariance matrix - historically
cov_matrix = returns.cov() * 252

# CASE I : Equally weighted portfolio using hist. mean and variance
# Portfolio's historical returns and risk assuming equal weights
print('I. Equally-weighted portfolio')
w = np.array([1 / len(tickers)] * len(tickers))
portfolio_eq_var = w.T @ cov_matrix @ w
portfolio_eq_std = np.sqrt(portfolio_eq_var)
portfolio_eq_return = returns.mean().mean() * 252
print(
    '''The equally weighted portfolio had return {:0.4f} and volatility equal to {:0.4f}.'''.format(portfolio_eq_return,
                                                                                                    portfolio_eq_std))


# VaR
def cvar(ret=returns, w=w):
    por_rets = (ret * w).sum(axis=1)
    var_hist = por_rets.quantile(0.05)
    cvar_hist = por_rets[por_rets <= var_hist].mean()
    print("VaR: {:.2f}%".format(100 * var_hist))
    print("CVaR: {:.2f}%".format(100 * cvar_hist))
    print('This value of the CVaR means that our average loss on the worst 5% of days will be {:.2f}%.'.format(
        cvar_hist * 100))


cvar()

# We find the expected return using the exponential moving average,
# which is a simple improvement over the mean historical return
exp_returns_est = expected_returns.ema_historical_return(df, span=1000)  # span was 500 by default

# Estimate the cov matrix using Ledoit-Wolf shrinkage
cov_matrix_est = risk_models.CovarianceShrinkage(df).ledoit_wolf(shrinkage_target='constant_variance')

# CASE II : CVaR portfolio, long positions only
# maximise return subject to a CVaR constraint
print('\nII. CVaR optimized, long positions only')
CVaR = EfficientCVaR(exp_returns_est, returns.dropna(), beta=0.95, weight_bounds=(0, 1), solver=None, verbose=False,
                     solver_options=None)
op_cvar = CVaR.efficient_risk(target_cvar=0.02)
w_cvar = CVaR.clean_weights()

# plot and save the weights
pd.Series(w_cvar).plot.pie(figsize=(10, 10),ylabel='ETFs', title='Weights for CvaR optimized portfolio')
plt.savefig('ii_cvar.png')

# Get the discrete allocation
def allocation(w, df=df, value=pf_value):
    da = DiscreteAllocation(w, get_latest_prices(df), total_portfolio_value=value)
    alloc, leftover = da.lp_portfolio()
    print('Allocation: ', alloc)
    print('Leftover: ', leftover)


allocation(w_cvar)
print(CVaR.portfolio_performance(verbose=True));
portfolio_cvar_vol = np.array(list(w_cvar.values())).T @ cov_matrix_est @ np.array(list(w_cvar.values()))
print('Annual Volatility:{:0.2f}% '.format(np.sqrt(portfolio_cvar_vol) * 100))

# CASE III : Global-Minimum Variance Portfolio
# research demonstrates GMV portfolios outperform mean-variance optimized portfolios
print('\nIII. GMV optimized')
ef = EfficientFrontier(exp_returns_est, cov_matrix_est, weight_bounds=(0, 1))
ef.min_volatility()
weights = ef.clean_weights()
allocation(weights)
print(ef.portfolio_performance(verbose=True))

# CASE IV : Max Return for a given risk
# research demonstrates GMV portfolios outperform mean-variance optimized portfolios
# ou are much more likely to get better results by enforcing some level of diversification.
# One way of doing this is to use L2 regularisation – essentially, adding a penalty on
# the number of near-zero weights.
print('\nIV. Max Return for a given risk')
ef = EfficientFrontier(exp_returns_est, cov_matrix_est, weight_bounds=(0, 1))
ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamma is the tuning parameter
ef.efficient_risk(target_volatility=0.08)
weights = ef.clean_weights()
allocation(weights)
print(ef.portfolio_performance(verbose=True))
pd.Series(weights).plot.pie(figsize=(10, 10),ylabel='ETFs', title='Weights for Max Return portfolio, given fixed risk')
plt.savefig('iv_max_return.png')

# CASE V : Max Sharpe
print('\nV. Max Sharpe')
ef = EfficientFrontier(exp_returns_est, cov_matrix_est, weight_bounds=(0, 1))
# ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamma is the tuning parameter
ef.max_sharpe(risk_free_rate=0.02)
weights = ef.clean_weights()
allocation(weights)
print(ef.portfolio_performance(verbose=True))

sys.stdout.close()
