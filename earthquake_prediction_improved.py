import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

# Find optimal DBSCAN parameters using a validation approach
def find_optimal_dbscan_params(X, min_samples_range=[5, 10, 15], eps_range=[0.05, 0.1, 0.15, 0.2]):
    best_score = -1
    best_params = {}
    
    for min_samples in min_samples_range:
        for eps in eps_range:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            n_noise = list(db.labels_).count(-1)
            
            # Calculate silhouette score if we have more than 1 cluster and less than 95% noise
            if n_clusters > 1 and n_noise / len(X) < 0.95:
                from sklearn.metrics import silhouette_score
                try:
                    score = silhouette_score(X, db.labels_)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                except:
                    continue
    
    return best_params if best_params else {'eps': 0.1, 'min_samples': 10}

# Find optimal DBSCAN parameters
optimal_params = find_optimal_dbscan_params(coords)
print(f"Optimal DBSCAN params: {optimal_params}")

# Apply DBSCAN with optimal parameters
db = DBSCAN(eps=optimal_params['eps'], min_samples=optimal_params['min_samples']).fit(coords)
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

# 7. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Hyperparameter Tuning for XGBoost
print("Performing hyperparameter tuning...")

# Define parameter grid for XGBoost - smaller grid to reduce computation time
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3]
}

# Calculate scale_pos_weight for handling class imbalance
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

# Create XGBoost model with initial parameters
xgb_model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=ratio,
    random_state=42
)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# 9. Train the best model and make predictions
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)
pred_proba = best_model.predict_proba(X_test)[:, 1]

# 10. Evaluate the improved model
accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, pred_proba)

print(f"Improved Model Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("\nDetailed Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# 11. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 12. Save the improved model and scaler
joblib.dump(best_model, 'earthquake_model_improved.pkl')
joblib.dump(scaler, 'scaler.pkl')
df.to_csv('processed_earthquake_data_improved.csv', index=False)
print("\nImproved model and files saved successfully.")