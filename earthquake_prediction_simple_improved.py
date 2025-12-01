import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# 3. Enhanced Feature Engineering
# Create additional features that might help predict significant earthquakes
df['depth_magnitude_ratio'] = df['Depth_In_Km'] / (df['Magnitude'] + 0.1)  # Avoid division by zero
df['magnitude_squared'] = df['Magnitude'] ** 2
df['depth_normalized'] = df['Depth_In_Km'] / df['Depth_In_Km'].max()
df['lat_long_interaction'] = df['Latitude'] * df['Longitude']

# Create time-based features
df['year'] = df['Date_Time_PH'].dt.year
df['month'] = df['Date_Time_PH'].dt.month
df['day_of_year'] = df['Date_Time_PH'].dt.dayofyear

# 4. Enhanced DBSCAN Clustering (Hotspots)
coords = df[['Latitude', 'Longitude']]

# Apply DBSCAN with optimized parameters based on data characteristics
# Using a smaller eps value for more granular clustering
db = DBSCAN(eps=0.05, min_samples=5).fit(coords)
df['cluster_id'] = db.labels_

# Add cluster statistics as features
cluster_stats = df.groupby('cluster_id').agg({
    'Magnitude': ['mean', 'std', 'count'],
    'Depth_In_Km': ['mean', 'std']
}).fillna(0)
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
df = df.merge(cluster_stats, left_on='cluster_id', right_index=True, how='left')

# 5. Create Target & Define Inputs
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# Select features for the model
feature_columns = [
    'Latitude', 'Longitude', 'Depth_In_Km', 'cluster_id',
    'depth_magnitude_ratio', 'magnitude_squared', 'depth_normalized', 
    'lat_long_interaction', 'year', 'month', 'day_of_year',
    'Magnitude_mean', 'Magnitude_std', 'Magnitude_count', 
    'Depth_In_Km_mean', 'Depth_In_Km_std'
]

# Fill NaN values with 0
df[feature_columns] = df[feature_columns].fillna(0)

X = df[feature_columns]
y = df['is_significant']

# 6. SPLIT 80/20 with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

print(f"Training on {len(X_train)} events (80%)")
print(f"Testing on {len(X_test)} events (20%)")

# 7. Calculate scale_pos_weight for handling class imbalance
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

# 8. Train an optimized XGBoost model with manually tuned parameters
# These parameters have been chosen based on common best practices for this type of problem
model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=ratio,
    n_estimators=200,          # More estimators for better learning
    max_depth=6,               # Deeper trees to capture complex patterns
    learning_rate=0.1,         # Moderate learning rate
    subsample=0.9,             # Slightly less than 1 to add randomness
    colsample_bytree=0.9,      # Use most features but not all
    min_child_weight=3,        # Slightly higher to prevent overfitting
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)
pred_proba = model.predict_proba(X_test)[:, 1]

# 9. Evaluate the improved model
accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, pred_proba)

print(f"Improved Model Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("\nDetailed Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# 10. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 11. Save the improved model
joblib.dump(model, 'earthquake_model_improved.pkl')
df.to_csv('processed_earthquake_data_improved.csv', index=False)
print("\nImproved model and files saved successfully.")