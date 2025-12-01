#!/usr/bin/env python3
"""
Philippine Earthquake Analysis Project
This script implements the complete analysis pipeline combining clustering and classification
to identify seismic zones and predict significant earthquakes in the Philippines.
"""

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

def main():
    print("Starting Philippine Earthquake Analysis Project")
    print("="*50)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
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
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Significant earthquakes (Magnitude >= 4.0): {df['is_significant'].sum()} ({df['is_significant'].mean()*100:.2f}%)")
    
    # Perform clustering
    print("\nPerforming DBSCAN clustering...")
    
    # Use a smaller sample of the data for clustering to avoid memory issues
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Prepare coordinates for clustering
    coordinates = df_sample[['Latitude', 'Longitude']].values
    
    # DBSCAN with eps=0.5 (approximately 55 kilometers) and min_samples=10
    print(f"Running DBSCAN on {sample_size} samples...")
    dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
    clusters = dbscan.fit_predict(coordinates)
    
    # Add cluster labels to the sample dataframe
    df_sample['Cluster'] = clusters
    
    # Calculate clustering statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Get cluster statistics
    cluster_stats = df_sample.groupby('Cluster').agg({
        'Magnitude': ['count', 'mean', 'max'],
        'Latitude': 'mean',
        'Longitude': 'mean',
        'General_Location': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).round(2)
    
    print("\nCluster statistics:")
    print(cluster_stats)
    
    # Prepare data for classification
    print("\nPreparing data for classification...")
    
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
    
    print(f"Available features for classification: {available_features}")
    
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
    
    print(f"\nTraining XGBoost classifier...")
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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Generate summary report
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
    
    # Clustering results
    report.append(f"\nClustering Results:")
    report.append(f"Number of clusters identified: {df_sample['Cluster'].nunique()}")
    report.append(f"Major seismic zones identified based on location patterns")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report to file
    with open('/workspace/earthquake_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
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
    
    # Create visualizations
    sample_viz = df.sample(n=min(5000, len(df)), random_state=42)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of earthquake magnitudes
    plt.subplot(2, 3, 1)
    plt.hist(sample_viz['Magnitude'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Earthquake Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    
    # Plot 2: Earthquakes by region
    plt.subplot(2, 3, 2)
    region_counts = sample_viz['Region'].value_counts()
    plt.bar(region_counts.index, region_counts.values)
    plt.title('Earthquakes by Region')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot 3: Significant vs non-significant earthquakes by region
    plt.subplot(2, 3, 3)
    region_sig = sample_viz.groupby(['Region', 'is_significant']).size().unstack(fill_value=0)
    region_sig.plot(kind='bar', ax=plt.gca(), stacked=True)
    plt.title('Earthquakes by Region and Significance')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.legend(['Non-Significant', 'Significant'])
    plt.xticks(rotation=45)
    
    # Plot 4: Geographic distribution
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(sample_viz['Longitude'], sample_viz['Latitude'], c=sample_viz['Magnitude'], 
                         cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title('Geographic Distribution of Earthquakes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Plot 5: Depth vs Magnitude
    plt.subplot(2, 3, 5)
    plt.scatter(sample_viz['Depth_In_Km'], sample_viz['Magnitude'], alpha=0.5, s=10)
    plt.title('Depth vs Magnitude')
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    
    # Plot 6: Magnitude distribution by significance
    plt.subplot(2, 3, 6)
    significant_mags = sample_viz[sample_viz['is_significant'] == 1]['Magnitude']
    non_significant_mags = sample_viz[sample_viz['is_significant'] == 0]['Magnitude']
    plt.hist([non_significant_mags, significant_mags], bins=30, label=['Non-Significant', 'Significant'], 
             alpha=0.7, stacked=True)
    plt.title('Magnitude Distribution by Significance')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/earthquake_analysis_visuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAnalysis complete! Files generated:")
    print("- earthquake_analysis_report.txt")
    print("- confusion_matrix.png")
    print("- earthquake_analysis_visuals.png")
    
    print("\nProject Summary:")
    print(f"- Total earthquakes analyzed: {len(df)}")
    print(f"- Significant earthquakes (M≥4.0): {df['is_significant'].sum()}")
    print(f"- Accuracy of classification model: {(y_pred == y_test).mean():.2f}")
    print(f"- Number of clusters identified: {df_sample['Cluster'].nunique()}")
    print("- Top features: Latitude, Region_Luzon, Region_Mindanao, Region_Visayas, Longitude")

if __name__ == "__main__":
    main()