# Philippine Earthquake Analysis Project

## Project Overview
This project analyzes historical PHIVOLCS earthquake data to identify persistent seismic zones and predict high-impact seismic events (Magnitude ≥ 4.0) using machine learning techniques. The analysis combines unsupervised learning (DBSCAN clustering) to identify seismic zones and supervised learning (XGBoost classification) to predict significant earthquakes.

## Data Source
- Dataset: "Philippine Earthquakes from PHIVOLCS" from Kaggle
- Time Period: 2016 to 2025
- Records: 113,276 earthquakes after preprocessing
- Features: Date, Latitude, Longitude, Depth, Magnitude, Location, General_Location

## Methodology

### 1. Clustering (DBSCAN)
- Used to identify persistent seismic zones based on geographic proximity
- Parameters: eps=0.5 (approx. 55km), min_samples=10
- Identified 2 main clusters in the Philippines

### 2. Classification (XGBoost)
- Binary classification to predict if an earthquake is significant (Magnitude ≥ 4.0)
- Features used: Latitude, Longitude, Depth, Time features, and Region
- Addressed class imbalance using scale_pos_weight parameter
- Model optimized for precision and recall on the minority class

## Key Results

### Clustering Results
- **Number of clusters identified**: 2
- **Major seismic zones**: Identified based on location patterns
- **Most active areas**: Surigao Del Sur and Davao Occidental regions

### Classification Results
- **Overall accuracy**: 80%
- **Precision for significant quakes**: 12%
- **Recall for significant quakes**: 61%
- **F1-score for significant quakes**: 20%

### Feature Importance
1. Latitude (20.86%)
2. Region_Luzon (15.65%)
3. Region_Mindanao (14.93%)
4. Region_Visayas (12.39%)
5. Longitude (9.74%)

### Regional Breakdown
- **Mindanao**: 56,137 total quakes, 3,070 significant quakes (5.47%)
- **Luzon**: 33,787 total quakes, 887 significant quakes (2.63%)
- **Visayas**: 22,160 total quakes, 626 significant quakes (2.82%)
- **Unknown**: 1,192 total quakes, 34 significant quakes (2.85%)

### Top Provinces for Significant Earthquakes
1. Davao Occidental: 1,228 quakes
2. Surigao Del Sur: 641 quakes
3. Davao Oriental: 444 quakes
4. Surigao Del Norte: 273 quakes
5. Eastern Samar: 179 quakes

## Files Generated
- `earthquake_analysis_report.txt`: Complete analysis summary
- `earthquake_analysis_visuals.png`: Multiple visualizations of the data
- `confusion_matrix.png`: Confusion matrix for the classification model

## Practical Applications
- Supports Local Government Units (LGUs) in prioritizing disaster preparedness
- Helps DRRMOs focus resources on high-risk areas
- Aids PHIVOLCS in refining seismic hazard assessments
- Contributes to the National Disaster Risk Reduction and Management Plan (NDRRMP)

## Limitations
- Model has low precision for predicting significant quakes (12%) but good recall (61%)
- Results based on historical patterns and should not be used for exact predictions
- Regional classification may have some misclassifications due to complex geography

## Conclusion
The analysis successfully identified key seismic zones in the Philippines, particularly in Mindanao which shows the highest rate of significant earthquakes. The XGBoost model provides a data-driven approach to flag potentially significant earthquakes for further review by seismologists.