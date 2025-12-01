import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb

# Load the data
df = pd.read_csv('/workspace/phivolcs_earthquake_data.csv')

# Preprocess data
df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
cutoff_date = pd.to_datetime('2025-10-15')
df = df[df['Date_Time_PH'] <= cutoff_date]

# Convert Depth_In_Km to numeric, handling any non-numeric values
df['Depth_In_Km'] = pd.to_numeric(df['Depth_In_Km'], errors='coerce')

# Drop rows with missing coordinates, magnitude, or depth
df = df.dropna(subset=['Latitude', 'Longitude', 'Magnitude', 'Depth_In_Km'])

# Create binary target variable: is_significant (Magnitude >= 4.0)
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# Parse General_Location into broader regions (Luzon, Visayas, Mindanao)
def classify_region(location):
    if pd.isna(location):
        return 'Unknown'
    
    location_lower = location.lower()
    
    # Luzon regions
    luzon_regions = ['bulacan', 'pampanga', 'bataan', 'angeles', 'baguio', 'benguet', 'tarlac', 
                    'nueva ecija', 'zambales', 'pangasinan', 'la union', 'ilocos', 'abracan',
                    'abra', 'apayao', 'benguet', 'ifugao', 'kalinga', 'mountain province',
                    'cagayan', 'isabela', 'nueva vizcaya', 'quirino', 'batanes', 'basco',
                    'rizon', 'marinduque', 'romblon', 'quezon', 'bicol', 'albay', 'camarines',
                    'catanduanes', 'masbate', 'sorsogon', 'aurora', 'bataan', 'bulacan',
                    'nueva ecija', 'pampanga', 'tarlac', 'zambales', 'manila', 'quezon city',
                    'makati', 'pasig', 'mandaluyong', 'san juan', 'caloocan', 'malabon',
                    'navotas', 'valenzuela', 'marikina', 'pasay', 'paranaque', 'muntinlupa',
                    'las pinas', 'makati', 'taguig', 'pateros', 'cavite', 'laguna', 'batangas',
                    'rizal', 'cavite', 'laguna', 'batangas', 'quezon', 'marinduque']
    
    # Visayas regions
    visayas_regions = ['cebu', 'cebu city', 'negros', 'negros oriental', 'negros occidental',
                      'bohol', 'siquijor', 'biliran', 'leyte', 'southern leyte', 'western samar',
                      'eastern samar', 'northern samar', 'samara', 'iloilo', 'capiz', 'antique',
                      'guimaras', 'aclar', 'palawan', 'romblon', 'mindoro', 'marinduque',
                      'panay', 'masbate', 'catanduanes', 'sorsogon']
    
    # Mindanao regions
    mindanao_regions = ['davao', 'davao del sur', 'davao del norte', 'davao oriental', 'davao occidental',
                       'cotabato', 'north cotabato', 'south cotabato', 'sultan kudarat', 'sarangani',
                       'general santos', 'butuan', 'surigao', 'surigao del norte', 'surigao del sur',
                       'agusan', 'agusan del norte', 'agusan del sur', 'misamis', 'misamis oriental',
                       'misamis occidental', 'camiguin', 'zamboanga', 'zamboanga del norte',
                       'zamboanga del sur', 'zamboanga sibugay', 'lanao', 'lanao del norte',
                       'lanao del sur', 'maguindanao', 'cagayan de oro', 'iligan', 'marawi',
                       'malaybalay', 'valencia', 'dipolog', 'roxas', 'bukidnon', 'camiguin',
                       'misamis', 'tawi-tawi', 'sulu', 'basilan']
    
    for region in luzon_regions:
        if region in location_lower:
            return 'Luzon'
    
    for region in visayas_regions:
        if region in location_lower:
            return 'Visayas'
    
    for region in mindanao_regions:
        if region in location_lower:
            return 'Mindanao'
    
    return 'Unknown'

df['Region'] = df['General_Location'].apply(classify_region)

# Create additional features for classification
df['Hour'] = df['Date_Time_PH'].dt.hour
df['DayOfWeek'] = df['Date_Time_PH'].dt.dayofweek
df['Month'] = df['Date_Time_PH'].dt.month

# Prepare features for classification
feature_columns = ['Latitude', 'Longitude', 'Depth_In_Km', 'Hour', 'DayOfWeek', 'Month', 'Region']

# Encode categorical features
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['Region'], prefix=['Region'])

# Select features that exist in the dataset
available_features = []
for col in feature_columns:
    if col == 'Region':
        # Add all region dummy columns
        region_cols = [c for c in df_encoded.columns if c.startswith('Region_')]
        available_features.extend(region_cols)
    elif col in df_encoded.columns:
        available_features.append(col)

X = df_encoded[available_features]
y = df_encoded['is_significant']

# Fill any remaining NaN values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate scale_pos_weight to handle class imbalance
neg_count = len(y_train[y_train == 0])
pos_count = len(y_train[y_train == 1])
scale_pos_weight = neg_count / pos_count

# Create and train XGBoost classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Significant', 'Significant'],
            yticklabels=['Non-Significant', 'Significant'])
plt.title('Confusion Matrix - XGBoost Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('/workspace/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("Confusion matrix visualization created successfully!")