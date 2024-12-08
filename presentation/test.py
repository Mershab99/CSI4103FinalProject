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
