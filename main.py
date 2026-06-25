# -*- coding: utf-8 -*-
"""
Semiconductor Supply Chain Shock Absorber Model
Author: Julian Salehi
"""
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime

def fetch_macro_data(series_id='PCU3332423332420', start_date='2010-01-01'):
    print(f"Fetching macro data ({series_id}) from FRED...")
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    df_ppi = web.DataReader(series_id, 'fred', start_date, end_date)
    df_ppi.rename(columns={series_id: 'Semi_PPI'}, inplace=True)
    return df_ppi

def fetch_equity_data(start_date, end_date):
    print("Fetching equity data from Yahoo Finance...")
    tickers = ["^GSPC", "^YH31130010"]
    df_equity = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    df_equity.rename(columns={'^GSPC': 'SP500', '^YH31130010': 'Semi_Index'}, inplace=True)
    if df_equity['Semi_Index'].isnull().all():
        df_equity['Semi_Index'] = yf.download("SMH", start=start_date, end=end_date)['Adj Close']
    return df_equity

def process_and_merge_data(df_ppi, df_equity, lag_months=3):
    print("Processing and aligning datasets...")
    df_ppi_processed = df_ppi.copy()
    df_ppi_processed['PPI_Shock_YoY'] = df_ppi_processed['Semi_PPI'].pct_change(periods=12)
    df_ppi_processed = df_ppi_processed.resample('ME').last() 
    df_equity_monthly = df_equity.resample('ME').last().pct_change()
    df_merged = pd.merge(df_equity_monthly, df_ppi_processed[['PPI_Shock_YoY']], left_index=True, right_index=True, how='inner')
    df_merged[f'PPI_Shock_Lag_{lag_months}M'] = df_merged['PPI_Shock_YoY'].shift(lag_months)
    return df_merged.dropna()

def run_regression_model(df_merged, lag_months=3):
    print("\n--- Supply Chain Shock Absorber Regression Results ---")
    X = df_merged[['SP500', f'PPI_Shock_Lag_{lag_months}M']]
    X = sm.add_constant(X)
    y = df_merged['Semi_Index']
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model

def generate_visualizations(df_equity, df_ppi, df_merged, lag_months=3, rolling_window=24):
    print("Generating visualizations...")
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    color = 'tab:blue'
    axes[0].set_ylabel('Semi Index Level', color=color)
    axes[0].plot(df_equity.index, df_equity['Semi_Index'], color=color, label='Semi Index')
    axes[0].tick_params(axis='y', labelcolor=color)

    ax2 = axes[0].twinx()
    color = 'tab:red'
    ax2.set_ylabel('Semiconductor Machinery PPI (Cost)', color=color)
    ax2.plot(df_ppi.index, df_ppi['Semi_PPI'], color=color, linestyle='--', label='PPI Cost')
    ax2.tick_params(axis='y', labelcolor=color)
    axes[0].set_title('Absolute Levels: Semiconductor Equities vs. Manufacturing PPI')

    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].plot(df_merged.index, df_merged['Semi_Index'], label='Semi Monthly Return', alpha=0.6)
    axes[1].plot(df_merged.index, df_merged[f'PPI_Shock_Lag_{lag_months}M'], label=f'{lag_months}M Lagged PPI Shock (YoY)', color='red', linewidth=2)
    axes[1].set_title('Rate of Change: Supply Shocks vs Equity Returns')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    df_ppi = fetch_macro_data()
    start_date = df_ppi.index.min().strftime('%Y-%m-%d')
    end_date = df_ppi.index.max().strftime('%Y-%m-%d')
    df_equity = fetch_equity_data(start_date, end_date)
    lag_months = 3
    df_merged = process_and_merge_data(df_ppi, df_equity, lag_months=lag_months)
    run_regression_model(df_merged, lag_months=lag_months)
    generate_visualizations(df_equity, df_ppi, df_merged, lag_months=lag_months)

if __name__ == "__main__":
    main()
