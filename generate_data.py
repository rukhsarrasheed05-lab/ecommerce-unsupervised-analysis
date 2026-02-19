"""
generate_data.py
----------------
Generates a synthetic e-commerce dataset for demonstration.
Run once: python generate_data.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

N_CUSTOMERS = 500
N_PRODUCTS  = 50
N_RATINGS   = 3000

# ── Customer Demographics & Behaviour ────────────────────────────────────────
age        = np.random.randint(18, 70, N_CUSTOMERS)
annual_inc = np.random.normal(55_000, 20_000, N_CUSTOMERS).clip(15_000, 200_000)

# 4 latent segments: budget, mid, premium, whale
segments = np.random.choice([0,1,2,3], N_CUSTOMERS, p=[0.3,0.35,0.25,0.10])

# Spend influenced by segment
base_spend = {0:50, 1:200, 2:600, 3:2000}
monthly_spend = np.array([
    np.random.normal(base_spend[s], base_spend[s]*0.3)
    for s in segments
]).clip(10)

n_purchases  = np.round(monthly_spend / np.random.uniform(20,80,N_CUSTOMERS)).clip(1).astype(int)
recency_days = np.random.randint(1, 365, N_CUSTOMERS)
avg_order    = monthly_spend / n_purchases
sessions_pm  = np.random.poisson(n_purchases * 1.5 + 1)
return_rate  = np.random.beta(1.5, 8, N_CUSTOMERS)   # mostly low

# Inject ~3% anomalies
n_anom = int(N_CUSTOMERS * 0.03)
anom_idx = np.random.choice(N_CUSTOMERS, n_anom, replace=False)
monthly_spend[anom_idx] *= np.random.uniform(8, 15, n_anom)
n_purchases[anom_idx]   *= np.random.randint(5, 12, n_anom)
return_rate[anom_idx]    = np.random.uniform(0.6, 0.95, n_anom)

customers = pd.DataFrame({
    "customer_id": [f"C{str(i).zfill(4)}" for i in range(N_CUSTOMERS)],
    "age":          age,
    "annual_income": annual_inc.round(2),
    "monthly_spend": monthly_spend.round(2),
    "n_purchases":   n_purchases,
    "recency_days":  recency_days,
    "avg_order_value": avg_order.round(2),
    "sessions_per_month": sessions_pm,
    "return_rate":   return_rate.round(4),
    "segment_label": segments,          # for reference only
})

# ── Product Ratings (for Collaborative Filtering) ────────────────────────────
categories = ["Electronics","Clothing","Books","Sports","Home","Beauty","Toys","Food"]
products = pd.DataFrame({
    "product_id":   [f"P{str(i).zfill(3)}" for i in range(N_PRODUCTS)],
    "category":     np.random.choice(categories, N_PRODUCTS),
    "price":        np.random.lognormal(4, 1, N_PRODUCTS).round(2).clip(5, 500),
})

cust_ids = np.random.choice(customers.customer_id, N_RATINGS)
prod_ids = np.random.choice(products.product_id,   N_RATINGS)
ratings_val = np.clip(
    np.random.normal(3.5, 1.2, N_RATINGS), 1, 5
).round(1)

ratings = pd.DataFrame({
    "customer_id": cust_ids,
    "product_id":  prod_ids,
    "rating":      ratings_val,
    "timestamp":   pd.date_range("2023-01-01", periods=N_RATINGS, freq="2h"),
}).drop_duplicates(subset=["customer_id","product_id"])

customers.to_csv("data/customers.csv", index=False)
products.to_csv("data/products.csv",   index=False)
ratings.to_csv("data/ratings.csv",     index=False)
print(f"✓ customers.csv  → {len(customers)} rows")
print(f"✓ products.csv   → {len(products)} rows")
print(f"✓ ratings.csv    → {len(ratings)} rows")
