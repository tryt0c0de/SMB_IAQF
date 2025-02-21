import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nyse_data = pd.read_csv("Data/NYSE_universe.csv",header=[0,1])[2:]
nyse_data.set_index([('Unnamed: 0_level_0',                       'Unnamed: 0_level_1')],inplace=True)
nyse_data.index.name = "Date"
nyse_data.index = pd.to_datetime(nyse_data.index)
monthly_data = nyse_data.resample('M').last()
### Buiding the different dataframes:
# Size (market cap)
size_df = monthly_data.xs('CUR_MKT_CAP', axis=1, level=1)

# Market-to-book or book-to-market
mb_df = monthly_data.xs('MARKET_CAPITALIZATION_TO_BV', axis=1, level=1)
# If this is M/B, invert it to get B/M:
bm_df = 1 / mb_df

# (Optional) If you want to compute OP = EBIT / Book_Equity or EBIT / Assets
ebit_df = monthly_data.xs('EBIT', axis=1, level=1)
assets_df = monthly_data.xs('BS_TOT_ASSET', axis=1, level=1)
# For illustration, define OP = EBIT / Assets:
op_df = ebit_df / assets_df

# (Optional) For INV = asset growth, you need the previous month's assets
# We can shift by 1 to get last month's assets, then compute growth
assets_lag = assets_df.shift(1)
inv_df = (assets_df - assets_lag) / assets_lag  # approximate % growth in total assets

# (Optional) If you only have 'PX_LAST', convert to returns:
px_last_df = monthly_data.xs('PX_LAST', axis=1, level=1)
ret_df = px_last_df.pct_change()  # approximate monthly returns

def create_2x3_labels(size_series, style_series):
    """
    size_series: Series of size for one month, index = tickers
    style_series: Series of characteristic (e.g. B/M) for the same month, index = tickers
    Returns a Series of labels, e.g. 'S-L', 'S-M', 'S-H', 'B-L', etc.
    """
    # Drop missing data
    df = pd.concat([size_series, style_series], axis=1).dropna()
    if df.empty:
        return pd.Series(index=size_series.index, dtype='object')
    
    # Breakpoint for size (median)
    size_median = df.iloc[:, 0].median()
    # Breakpoints for style (30% and 70%):
    style_30 = df.iloc[:, 1].quantile(0.3)
    style_70 = df.iloc[:, 1].quantile(0.7)

    labels = []
    for ticker, row in df.iterrows():
        sz = row.iloc[0]
        st = row.iloc[1]
        size_label = 'S' if sz <= size_median else 'B'

        if st <= style_30:
            style_label = 'L'  # e.g. "Low" for B/M, or "Growth"
        elif st > style_70:
            style_label = 'H'  # e.g. "High" for B/M, or "Value"
        else:
            style_label = 'M'  # "Middle"
        
        labels.append(size_label + '-' + style_label)
    
    return pd.Series(labels, index=df.index)

def construct_2x3_portfolio_returns(size_df, style_df, ret_df):
    """
    Constructs 2x3 portfolios (S-L, S-M, S-H, B-L, B-M, B-H) using:
      - size_df   : DataFrame of market caps (index = dates, columns = tickers)
      - style_df  : DataFrame of the 'style' measure (e.g. B/M, OP, or INV)
      - ret_df    : DataFrame of returns (index = dates, columns = tickers)
      
    Returns
    -------
    portfolio_returns : DataFrame
        Index = dates, columns = ['S-L','S-M','S-H','B-L','B-M','B-H'].
        Each cell is the equal-weighted average return of that portfolio in that month.
        
    Notes
    -----
    - We use the data at date t to assign tickers to bins, 
      then we take the *next* month's returns (date t+1) to compute portfolio performance.
    - If you prefer value weighting, replace the `.mean()` with a weighted average 
      using `size_df.loc[date]` as weights.
    """
    port_cols = ['S-L','S-M','S-H','B-L','B-M','B-H']
    portfolio_returns = pd.DataFrame(index=ret_df.index, columns=port_cols, dtype=float)
    
    # We iterate over each date except the very last (since we need the next month)
    for i in range(len(ret_df.index) - 1):
        date = ret_df.index[i]
        next_month = ret_df.index[i+1]
        
        # Current month's size & style
        size_series = size_df.loc[date]
        style_series = style_df.loc[date]
        
        # Assign each stock to a bin
        labels = create_2x3_labels(size_series, style_series)
        
        # Next month's returns
        next_ret = ret_df.loc[next_month].dropna()
        
        # Align returns with labels (so we only group tickers that have labels)
        valid_tickers = labels.dropna().index.intersection(next_ret.index)
        grouped = next_ret[valid_tickers].groupby(labels[valid_tickers])
        
        # Equal-weighted average return in each bucket
        avg_ret = grouped.mean()
        
        # Store results
        for bucket in avg_ret.index:
            portfolio_returns.loc[next_month, bucket] = avg_ret.loc[bucket]
    return portfolio_returns.iloc[1:]


# B/M-based 2×3
portfolio_returns_BM = construct_2x3_portfolio_returns(size_df, bm_df, ret_df)

# OP-based 2×3
portfolio_returns_OP = construct_2x3_portfolio_returns(size_df, op_df, ret_df)


######### THIS ONE HAS TOO MANY NANS THAT IS WHY IT IS EXCLUDED IN THE FINAL CALCULATIONS
######### NEED TO GO AND GET MORE DATA FROM BLOOMBERG
# INV-based 2×3 
portfolio_returns_INV = construct_2x3_portfolio_returns(size_df, inv_df, ret_df)



# 4) SMB(B/M), SMB(OP), SMB(INV)
portfolio_returns_BM['SMB_BM'] = (
    (portfolio_returns_BM['S-L'] + portfolio_returns_BM['S-M'] + portfolio_returns_BM['S-H']) / 3
    - (portfolio_returns_BM['B-L'] + portfolio_returns_BM['B-M'] + portfolio_returns_BM['B-H']) / 3
)

portfolio_returns_OP['SMB_OP'] = (
    (portfolio_returns_OP['S-L'] + portfolio_returns_OP['S-M'] + portfolio_returns_OP['S-H']) / 3
    - (portfolio_returns_OP['B-L'] + portfolio_returns_OP['B-M'] + portfolio_returns_OP['B-H']) / 3
)
#TOO MANY NANS
portfolio_returns_INV['SMB_INV'] = (
    (portfolio_returns_INV['S-L'] + portfolio_returns_INV['S-M'] + portfolio_returns_INV['S-H']) / 3
    - (portfolio_returns_INV['B-L'] + portfolio_returns_INV['B-M'] + portfolio_returns_INV['B-H']) / 3
)

# 5) Final SMB factor = average of the three SMB components
smb_bm = portfolio_returns_BM['SMB_BM']
smb_op = portfolio_returns_OP['SMB_OP']
smb_inv = portfolio_returns_INV['SMB_INV']

smb = (smb_bm + smb_op) / 2


smb = smb.drop(pd.Timestamp('2022-02-28'))

cumul = (1+smb).cumprod()
mean_return = smb.mean()
sharpe_ratio = (smb.mean() / smb.std()) * np.sqrt(12) ## Since they are monthly returns
print("Mean Return per month:", mean_return)
print("Annualized Sharpe Ratio:", sharpe_ratio)
plt.grid()
plt.plot(cumul)
plt.show()