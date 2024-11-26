
---

# What is the S&P 500?

- The S&P 500 Index serves as a barometer for the movement of the U.S. equity market.
- It tracks the price movements of 500 leading U.S. companies, capturing the activity of approximately 80% of the market capitalization of all U.S. stocks.
- 
---

## Spectral Decomposition Investigation of S&P500

1. Gather Data:

```python

```

2. Clean Data:

```python
# Step 2: Preprocessing - Remove Excessive Missing Values
data_rem = data.dropna(thresh=int(0.8 * len(data.columns)), axis=0)  # At least 80% valid rows
data_rem = data_rem.dropna(thresh=int(0.8 * len(data)), axis=1)  # At least 80% valid columns
```
---

3. Calculate Centered log returns:

```python
# Step 3: Calculate centered log returns
centered_log_returns = log_returns - log_returns.mean()
```
4. Compute Covariance Matrix:

```python
# Step 4: Compute covariance matrix of centered log returns
cov_matrix = centered_log_returns.cov()
```
---

5. Compute Eigenvalues/Eigenvectors:

```python
# Step 5: Perform spectral decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```

- Sort eigenvalues and eigenvectors:

```python
# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
```

We can now evaluate the dominant principal component:
```python
# Dominant principal component
dominant_pc = eigenvectors[:, 0]
```

---

6. Compute Eigenvalues/Eigenvectors:

```python
# Step 5: Perform spectral decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```
---

