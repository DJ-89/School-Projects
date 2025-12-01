import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the earthquake dataset"""
    print("Loading earthquake dataset...")
    df = pd.read_csv('/workspace/phivolcs_earthquake_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Clean and preprocess the data according to project requirements"""
    print("\nPreprocessing data...")
    
    # Remove future-dated entries (beyond October 15, 2025)
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
    
    # Depth categories
    df['Depth_Category'] = pd.cut(df['Depth_In_Km'], 
                                  bins=[0, 10, 35, 70, 300, float('inf')], 
                                  labels=['Shallow', 'Intermediate', 'Deep', 'Very Deep', 'Extremely Deep'])
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Significant earthquakes (Magnitude >= 4.0): {df['is_significant'].sum()} ({df['is_significant'].mean()*100:.2f}%)")
    
    return df

def perform_clustering(df):
    """Perform DBSCAN clustering to identify persistent seismic zones"""
    print("\nPerforming DBSCAN clustering...")
    
    # Use a much smaller sample of the data for clustering to avoid memory issues
    # Take a random sample of the data
    sample_size = min(10000, len(df))  # Use maximum 10,000 samples
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Prepare coordinates for clustering
    coordinates = df_sample[['Latitude', 'Longitude']].values
    
    # DBSCAN with eps=0.5 (approximately 55 kilometers) and min_samples=10
    print(f"Running DBSCAN on {sample_size} samples...")
    dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
    clusters = dbscan.fit_predict(coordinates)
    
    # Add cluster labels to the sample dataframe
    df_sample['Cluster'] = clusters
    
    # Calculate silhouette score for clustering quality
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        try:
            silhouette_avg = silhouette_score(coordinates, clusters)
            print(f"Silhouette Score: {silhouette_avg:.3f}")
        except:
            print("Silhouette score calculation failed due to single cluster or other issue")
    else:
        print("Only one cluster found, silhouette score not applicable")
    
    # Get cluster statistics
    cluster_stats = df_sample.groupby('Cluster').agg({
        'Magnitude': ['count', 'mean', 'max'],
        'Latitude': 'mean',
        'Longitude': 'mean',
        'General_Location': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).round(2)
    
    print("\nCluster statistics:")
    print(cluster_stats)
    
    # Add cluster information back to main dataframe by assigning clusters based on nearest neighbors
    # This is a simplified approach - for each point in the full dataset, find the cluster of the nearest sample point
    from sklearn.neighbors import NearestNeighbors
    
    # Prepare the sample coordinates for nearest neighbor lookup
    sample_coordinates = df_sample[['Latitude', 'Longitude']].values
    
    # Prepare the full dataset coordinates (only for the rows that will be processed)
    full_coordinates = df[['Latitude', 'Longitude']].values
    
    # Fit nearest neighbors on the sample
    print("Assigning cluster labels to full dataset...")
    nbrs = NearestNeighbors(n_neighbors=1).fit(sample_coordinates)
    distances, indices = nbrs.kneighbors(full_coordinates)
    
    # Assign cluster labels based on nearest sample point
    df['Cluster'] = df_sample.iloc[indices.flatten()]['Cluster'].values
    
    return df, dbscan

def prepare_classification_data(df):
    """Prepare data for XGBoost classification"""
    print("\nPreparing data for classification...")
    
    # Select features for classification
    feature_columns = ['Latitude', 'Longitude', 'Depth_In_Km', 'Hour', 'DayOfWeek', 'Month', 'Region']
    
    # Encode categorical features
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['Region', 'Depth_Category'], prefix=['Region', 'Depth'])
    
    # Select features that exist in the dataset
    available_features = []
    for col in feature_columns:
        if col == 'Region':
            # Add all region dummy columns
            region_cols = [c for c in df_encoded.columns if c.startswith('Region_')]
            available_features.extend(region_cols)
        elif col == 'Depth_Category':
            # Add all depth dummy columns
            depth_cols = [c for c in df_encoded.columns if c.startswith('Depth_')]
            available_features.extend(depth_cols)
        elif col in df_encoded.columns:
            available_features.append(col)
    
    print(f"Available features for classification: {available_features}")
    
    X = df_encoded[available_features]
    y = df_encoded['is_significant']
    
    # Fill any remaining NaN values
    X = X.fillna(X.mean())
    
    return X, y, available_features

def train_xgboost_model(X, y):
    """Train XGBoost classifier to predict significant earthquakes"""
    print("\nTraining XGBoost classifier...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight to handle class imbalance
    neg_count = len(y_train[y_train == 0])
    pos_count = len(y_train[y_train == 1])
    scale_pos_weight = neg_count / pos_count
    
    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
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
    
    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - XGBoost Classifier')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('/workspace/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title('Top 10 Feature Importances - XGBoost Classifier')
    plt.tight_layout()
    plt.savefig('/workspace/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return xgb_model, X_test, y_test, y_pred, feature_importance

def visualize_results(df):
    """Create visualizations for the analysis results"""
    print("\nCreating visualizations...")
    
    # Use a sample for plotting to avoid memory issues
    sample_size = min(10000, len(df))
    df_viz = df.sample(n=sample_size, random_state=42)
    
    # Plot earthquake distribution by magnitude
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df_viz['Magnitude'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Earthquake Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    
    # Plot significant vs non-significant earthquakes by region
    plt.subplot(2, 2, 2)
    region_sig = df_viz.groupby(['Region', 'is_significant']).size().unstack(fill_value=0)
    region_sig.plot(kind='bar', stacked=True)
    plt.title('Earthquakes by Region and Significance')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.legend(['Non-Significant', 'Significant'])
    plt.xticks(rotation=45)
    
    # Geographic distribution of clusters (if clusters exist)
    plt.subplot(2, 2, 3)
    if 'Cluster' in df_viz.columns and df_viz['Cluster'].nunique() > 1:
        scatter = plt.scatter(df_viz['Longitude'], df_viz['Latitude'], c=df_viz['Cluster'], cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Earthquake Clusters (DBSCAN)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    else:
        plt.scatter(df_viz['Longitude'], df_viz['Latitude'], alpha=0.6)
        plt.title('Earthquake Locations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    # Depth vs Magnitude
    plt.subplot(2, 2, 4)
    plt.scatter(df_viz['Depth_In_Km'], df_viz['Magnitude'], alpha=0.5)
    plt.title('Depth vs Magnitude')
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.savefig('/workspace/earthquake_analysis_visuals.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory

def generate_summary_report(df, xgb_model, feature_importance):
    """Generate a summary report of findings"""
    print("\nGenerating summary report...")
    
    report = []
    report.append("Philippine Earthquake Analysis - Summary Report")
    report.append("="*50)
    
    report.append(f"\nTotal Earthquakes Analyzed: {len(df)}")
    report.append(f"Significant Earthquakes (Magnitude ≥ 4.0): {df['is_significant'].sum()}")
    report.append(f"Percentage of Significant Earthquakes: {df['is_significant'].mean()*100:.2f}%")
    
    # Top provinces with significant quakes
    significant_quakes = df[df['is_significant'] == 1]
    top_provinces = significant_quakes['General_Location'].value_counts().head(10)
    report.append(f"\nTop 10 Provinces with Most Significant Earthquakes:")
    for i, (province, count) in enumerate(top_provinces.items(), 1):
        report.append(f"{i}. {province}: {count} quakes")
    
    # Top features for prediction
    report.append(f"\nTop 5 Features for Predicting Significant Earthquakes:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        report.append(f"{i}. {row['feature']}: {row['importance']:.4f}")
    
    # Regional breakdown
    region_breakdown = df.groupby('Region')['is_significant'].agg(['count', 'sum', 'mean'])
    region_breakdown.columns = ['Total_Quakes', 'Significant_Quakes', 'Significance_Rate']
    region_breakdown['Significance_Rate'] = region_breakdown['Significance_Rate']*100
    report.append(f"\nEarthquake Distribution by Region:")
    for region in region_breakdown.index:
        row = region_breakdown.loc[region]
        report.append(f"{region}: {int(row['Total_Quakes'])} total quakes, "
                     f"{int(row['Significant_Quakes'])} significant quakes, "
                     f"{row['Significance_Rate']:.2f}% significance rate")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report to file
    with open('/workspace/earthquake_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text

def main():
    """Main function to execute the earthquake analysis"""
    print("Starting Philippine Earthquake Analysis Project")
    print("="*50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform clustering
    df, dbscan_model = perform_clustering(df)
    
    # Prepare data for classification
    X, y, feature_names = prepare_classification_data(df)
    
    # Train XGBoost model
    xgb_model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(X, y)
    
    # Create visualizations
    visualize_results(df)
    
    # Generate summary report
    report = generate_summary_report(df, xgb_model, feature_importance)
    
    print("\nAnalysis complete! Files generated:")
    print("- confusion_matrix.png")
    print("- feature_importance.png") 
    print("- earthquake_analysis_visuals.png")
    print("- earthquake_analysis_report.txt")
    
    return df, dbscan_model, xgb_model, feature_importance

if __name__ == "__main__":
    df, dbscan_model, xgb_model, feature_importance = main()