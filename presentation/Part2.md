---
author: "Mershab Issadien & Jenet Ghumman"
date: "MMMM" 
paging: Page %d of %d
---

# What is the S&P 500?

- The S&P 500 Index serves as a barometer for the movement of the U.S. equity market.

- It tracks the price movements of 500 leading U.S. companies, capturing the activity of approximately 80% of the market capitalization of all U.S. stocks.

**NOTE we can pick any set of stocks.**
---

## Spectral Decomposition Investigation of S&P500

# Step 1: Gather Data

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

# Download historical data for the last 5 years
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Adj Close']

```

# Step 2: Clean Data:

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

# Step 4: Compute Covariance Matrix

```python
# Formula: Cov(i, j) = E[(R_i - μ_i) * (R_j - μ_j)]
# The covariance matrix captures how pairs of stocks move together. Each element Cov(i, j) represents 
# the covariance between the log returns of stock i and stock j.
# Mathematically:
#   If R_i and R_j are the log returns of stocks i and j:
#     Cov(i, j) = Σ[(R_i[t] - μ_i) * (R_j[t] - μ_j)] / (n - 1)
#   Where:
#     μ_i = mean of log returns of stock i,
#     μ_j = mean of log returns of stock j,
#     n = number of observations.
cov_matrix = log_returns.cov()
```
---

# Step 5. Compute Eigenvalues/Eigenvectors:

```python
# Mathematically:
#   Cov_matrix = P * D * P^T
# Where:
#   P = matrix of eigenvectors (columns represent directions of principal components),
#   D = diagonal matrix of eigenvalues (magnitude of variance explained by each principal component),
#   P^T = transpose of P.
# np.linalg.eigh is used because the covariance matrix is symmetric and positive semi-definite.
# Eigenvalues represent the amount of variance explained by each principal component.
# Eigenvectors define the directions in the feature space corresponding to these components.
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```

##### Sort eigenvalues and eigenvectors:

```python
# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
```

---

# Analyze Dominant Principal Component

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

---

# Putting It All Together

```python
import yfinance as yf
import numpy as np
import pandas as pd

data = pd.read_csv("sp500_data.csv", index_col=0, parse_dates=True)

# Verify the data is loaded correctly
print("S&P 500 historical data successfully loaded.")

# Step 2: Preprocessing - Remove Excessive Missing Values
data_rem = data.dropna(thresh=int(0.8 * len(data.columns)), axis=0)  # At least 80% valid rows
data_rem = data_rem.dropna(thresh=int(0.8 * len(data)), axis=1)  # At least 80% valid columns

# Step 3: Calculate daily log returns
log_returns = np.log(data_rem / data_rem.shift(1)).dropna()

# %%
# Step 5: Compute covariance matrix
cov_matrix = log_returns.cov()

# Step 6: Perform spectral decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Explained variance ratio
explained_variance_ratio = eigenvalues / eigenvalues.sum()

# %%
# Step 7: Reconstruct stock price movements
# Choose the top k principal components
num_components = len(eigenvalues)  # Adjust to desired number of components, for now selecting all
principal_components = eigenvectors[:, :num_components]

# %%
# Step 10: Analyze the dominant principal component
dominant_pc = eigenvectors[:, 0]
# Ensure positive orientation of the dominant PC
if dominant_pc[0] < 0:
    dominant_pc *= -1

# Create a DataFrame for stock weights in the dominant principal component
stock_weights = pd.DataFrame({
    'Stock': log_returns.columns,
    'Weight': dominant_pc
}).sort_values(by='Weight', ascending=False)

# Output top 10 stocks by weight
print("Top 10 stocks by weight in the dominant principal component:")
print(stock_weights.head(10))
```

---

# Key Takeaways and Challenges

- Covariance matrix can be used to construct tailor made combo-orderbooks. (Elaborated in report.)
- Principal Component Analysis can be used to generate risk profiles and inform us about the most volatile stocks.

- We only learn correlation, **NOT Causation**. High Covariance or stocks with similar weights in a PC are not guaranteed to move as predicatably in the future.
    - Must be used as a starting point.
    - Reveals symptoms on the market, not cause.

---

# Future Directions

- Apply this methodology to other indices and asset classes.
    - Something we are exploring in our report. (ETFs, Commodities, Other composite asset classes)
- Develop a system to do this with real-time market data.
    - Rolling window, etc.

## Links and References:

- Public Project link: `https://github.com/Mershab99/CSI4103FinalProject`

- Presentation Gif Link: `https://github.com/Mershab99/CSI4103FinalProject/blob/main/presentation/presentation.gif`

Tools Used:

- Slides: `https://github.com/maaslalani/slides/tree/main`

- Presentation Gif: `https://github.com/charmbracelet/vhs`
