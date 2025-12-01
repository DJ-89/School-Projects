import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv('phivolcs_earthquake_data.csv')

# 2. Data Cleaning (Updated to fix crash)
# Convert Date
df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')

# Force numeric columns (fixes errors like '<001' or 'Unknown')
cols_to_clean = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that still have missing values
df = df.dropna(subset=cols_to_clean)

# 3. Create Target & Cluster
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# DBSCAN Clustering (Hotspots)
coords = df[['Latitude', 'Longitude']]
# eps=0.1 (~11km), min_samples=10
db = DBSCAN(eps=0.1, min_samples=10).fit(coords)
df['cluster_id'] = db.labels_

# 4. Define Inputs (X) and Target (y)
X = df[['Latitude', 'Longitude', 'Depth_In_Km', 'cluster_id']]
y = df['is_significant']

# 5. SPLIT 80/20
# test_size=0.2 means 20% for testing, leaving 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training on {len(X_train)} events (80%)")
print(f"Testing on {len(X_test)} events (20%)")

# 6. Train XGBoost
# scale_pos_weight helps the model pay attention to the rare 'Significant' quakes
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=ratio  # CRITICAL: Fixes the class imbalance
)
model.fit(X_train, y_train)

# 7. Evaluate
preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, preds))

# 8. Save
joblib.dump(model, 'earthquake_model.pkl')
df.to_csv('processed_earthquake_data.csv', index=False)
print("Files saved successfully.")