---
author: ""
date: ""
paging: Page %d of %d
---

# What is the S&P 500?

- The S&P 500 Index serves as a barometer for the movement of the U.S. equity market.
- It tracks the price movements of 500 leading U.S. companies, capturing the activity of approximately 80% of the market capitalization of all U.S. stocks.
---

## Spectral Decomposition Investigation of S&P500

1. Gather Data:

```python
# Step 1: Fetch S&P 500 data
def get_sp500_tickers():
    # Get S&P 500 tickers from Wikipedia
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(sp500_url, header=0)
    tickers = table[0]['Symbol'].tolist()
    return tickers

# Fetch S&P 500 tickers
tickers = get_sp500_tickers()
tickers = [ticker.replace('.', '-') for ticker in tickers]  # Adjust for yfinance

# Download historical data for the last year
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")['Adj Close']

```

2. Clean Data:

```python
# Step 2: Preprocessing - Remove Excessive Missing Values
data_rem = data.dropna(thresh=int(0.8 * len(data.columns)), axis=0)  # At least 80% valid rows
data_rem = data_rem.dropna(thresh=int(0.8 * len(data)), axis=1)  # At least 80% valid columns
```
---

# Step 3: Calculate log returns

```python
# Step 3: Calculate daily log returns
log_returns = np.log(data_rem / data_rem.shift(1)).dropna()
```

# Step 4: Mean-center the log returns

```python
centered_log_returns = log_returns - log_returns.mean()`
```

# Step 5: Compute Covariance Matrix

```python
# Step 5: Compute covariance matrix
cov_matrix = centered_log_returns.cov()
```
---

# Step 6. Compute Eigenvalues/Eigenvectors:

```python
# Step 6: Perform spectral decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```

- Sort eigenvalues and eigenvectors:

```python
# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


# Explained variance ratio
explained_variance_ratio = eigenvalues / eigenvalues.sum()
```

---

## Analyze Dominant Component

We can now evaluate the dominant principal component:
```python
# Step 10: Analyze the dominant principal component
dominant_pc = eigenvectors[:, 0]

# Ensure positive orientation of the dominant PC
if dominant_pc[0] < 0:
    dominant_pc *= -1

# Create a DataFrame for stock weights in the dominant principal component
stock_weights = pd.DataFrame({
    'Stock': centered_log_returns.columns,
    'Weight': dominant_pc
}).sort_values(by='Weight', ascending=False)

# Output top 10 stocks by weight
print("Top 10 stocks by weight in the dominant principal component:")
print(stock_weights.head(10))
```
