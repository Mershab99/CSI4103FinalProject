import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Acquisition
symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]  # Sample S&P 500 tickers
sp500_data = yf.download(symbols, start="2021-01-01", end="2024-01-01")

# Extract relevant features
closing_price = sp500_data["Close"]
volume = sp500_data["Volume"]
market_cap = closing_price * volume  # Approximation for market cap

# Combine features into a single DataFrame
features_data = pd.concat([closing_price, volume, market_cap], axis=1, keys=["Close", "Volume", "Market Cap"])

# 2. Data Processing
# Calculate daily percentage changes for all features
daily_pct_changes = features_data.pct_change().dropna()

# 3. Compute the covariance matrix
cov_matrix = daily_pct_changes.cov()
print("Covariance Matrix:")
print(cov_matrix)

# 4. Perform PCA
# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Display eigenvalues and eigenvectors
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Identify the principal components
explained_variance_ratio = eigenvalues / sum(eigenvalues)
print("\nExplained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i+1}: {ratio:.2%}")

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print("\nCumulative Explained Variance:")
print(cumulative_variance)

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7, label="Individual Explained Variance")
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', label="Cumulative Explained Variance", linestyle='--', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.show()

# 6. Tying Explained Variance Ratios to Features
print("\nFeature Contribution to Principal Components:")

# Eigenvectors describe the contribution of each original feature to the principal components
feature_names = ["Close", "Volume", "Market Cap"]

for i in range(len(eigenvectors)):
    print(f"Principal Component {i+1}:")
    for j, feature in enumerate(feature_names):
        contribution = eigenvectors[j, i]
        print(f"  {feature}: {contribution:.4f}")

# Interpretation of explained variance
print("\nInterpretation:")
print("Each principal component is a weighted combination of the original features. For instance, a high weight for 'Market Cap' \n" +
      "in Principal Component 1 suggests that market cap heavily influences the variance captured by this component.")
print("The explained variance ratio shows how much of the total variance in the dataset is captured by each component, \n" +
      "helping to identify the most important patterns across 'Close', 'Volume', and 'Market Cap'.")


# Plotting eigenvalues (explained variance by principal component)
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(eigenvalues) + 1), explained_variance_ratio * 100, alpha=0.7, label="Explained Variance")
plt.ylabel("Explained Variance (%)")
plt.xlabel("Principal Component")
plt.title("Explained Variance by Principal Component")
plt.xticks(range(1, len(eigenvalues) + 1))
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()
#plt.show()
plt.savefig('explained_variance.png')

# Plotting eigenvectors (feature contributions to principal components)
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.plot(range(1, len(eigenvalues) + 1), eigenvectors[i], marker='o', label=f"{feature}")

plt.ylabel("Feature Contribution")
plt.xlabel("Principal Component")
plt.title("Feature Contributions to Principal Components (Eigenvectors)")
plt.xticks(range(1, len(eigenvalues) + 1))
plt.axhline(0, color="black", linewidth=0.7, linestyle="--")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
#plt.show()
plt.savefig('eigenvectors.png')
