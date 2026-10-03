"""Generate realistic sample datasets for testing."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_sales_dataset(n=1000):
    dates = [datetime(2022, 1, 1) + timedelta(days=i % 365 + np.random.randint(0, 10)) for i in range(n)]
    df = pd.DataFrame({
        "date": sorted(dates),
        "product": np.random.choice(["Laptop", "Phone", "Tablet", "Headphones", "Smartwatch"], n),
        "region": np.random.choice(["North", "South", "East", "West", "Central"], n),
        "sales_rep": np.random.choice([f"Rep_{i}" for i in range(1, 21)], n),
        "units_sold": np.random.randint(1, 50, n),
        "unit_price": np.round(np.random.uniform(100, 2000, n), 2),
        "discount_pct": np.round(np.random.choice([0, 5, 10, 15, 20], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]), 1),
        "customer_age": np.random.randint(18, 70, n),
        "customer_satisfaction": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        "return_flag": np.random.choice([0, 1], n, p=[0.92, 0.08]),
    })
    # Introduce some missing values
    for col in ["discount_pct", "customer_satisfaction"]:
        mask = np.random.random(n) < 0.05
        df.loc[mask, col] = np.nan
    df["revenue"] = df["units_sold"] * df["unit_price"] * (1 - df["discount_pct"].fillna(0) / 100)
    df["revenue"] = np.round(df["revenue"], 2)
    return df

if __name__ == "__main__":
    df = generate_sales_dataset(1000)
    df.to_csv("sales_data.csv", index=False)
    df.to_excel("sales_data.xlsx", index=False)
    print(f"Generated sales dataset: {df.shape}")
    print(df.dtypes)
    print(df.head())
