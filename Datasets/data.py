import yfinance as yf
import pandas as pd

stocks = {
    "Infosys": "INFY.NS",
    "HDFC_Bank": "HDFCBANK.NS",
    "Sun_Pharma": "SUNPHARMA.NS",
    "Reliance": "RELIANCE.NS",
    "Maruti": "MARUTI.NS"
}

for name, symbol in stocks.items():
    df = yf.download(
        symbol,
        start="2020-01-01",
        end="2025-01-01"
    )

    df.to_csv(f"Datasets/{name}.csv")

    print(f"{name} saved successfully")