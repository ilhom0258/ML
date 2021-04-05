import pandas as pd
import quandl, math
import numpy as np
import sklearn

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df["HL_PCT"] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Open', 'Adj. Volume']]

forecast_column = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())