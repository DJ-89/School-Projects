import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a sample earthquake dataset similar to what the script expects
np.random.seed(42)
n_records = 1000

# Generate sample data
days_offset = np.random.randint(0, 365*3, n_records)  # 3 years of data
data = {
    'Date_Time_PH': [
        (datetime.now() - timedelta(days=int(x))).strftime('%Y-%m-%d %H:%M:%S') 
        for x in days_offset
    ],
    'Latitude': np.random.uniform(4.0, 20.0, n_records),  # Philippines latitude range
    'Longitude': np.random.uniform(116.0, 127.0, n_records),  # Philippines longitude range
    'Depth_In_Km': np.random.uniform(1.0, 100.0, n_records),
    'Magnitude': np.random.gamma(2, 1.5, n_records) + 3  # Most magnitudes around 3-7 range
}

df = pd.DataFrame(data)

# Ensure some realistic constraints
df.loc[df['Magnitude'] > 9, 'Magnitude'] = np.random.uniform(7, 9, (df['Magnitude'] > 9).sum())

# Add some cluster-like patterns by occasionally generating coordinates close together
for i in range(0, n_records, 50):  # Every 50th record, create a cluster
    base_lat = np.random.uniform(8.0, 18.0)
    base_lon = np.random.uniform(120.0, 126.0)
    cluster_size = np.random.randint(5, 15)
    if i + cluster_size < n_records:
        df.loc[i:i+cluster_size-1, 'Latitude'] = np.random.normal(base_lat, 0.1, cluster_size)
        df.loc[i:i+cluster_size-1, 'Longitude'] = np.random.normal(base_lon, 0.1, cluster_size)

df.to_csv('phivolcs_earthquake_data.csv', index=False)
print(f"Created sample earthquake data with {len(df)} records")
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData info:")
print(df.info())