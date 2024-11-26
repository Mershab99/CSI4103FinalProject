import pandas as pd

# Sample data: Replace with real-world data for accuracy
data = {
    "Company": ["Company A", "Company B", "Company C"],
    "Share Price": [150, 120, 100],       # Prices of each share
    "Shares Outstanding": [1_000_000, 800_000, 500_000], # Total shares
    "Free Float Factor": [0.85, 0.90, 0.75]  # Fraction of shares available for public trading
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate free-float adjusted market capitalization
df["Free Float Market Cap"] = df["Share Price"] * df["Shares Outstanding"] * df["Free Float Factor"]

# Calculate total free-float market capitalization (index denominator)
total_market_cap = df["Free Float Market Cap"].sum()

# Calculate the weight of each company in the index
df["Index Weight"] = df["Free Float Market Cap"] / total_market_cap

# Display results
print(df[["Company", "Free Float Market Cap", "Index Weight"]])
