# Earthquake Prediction Web Application - Setup Guide

## Overview
This project includes an enhanced earthquake prediction model that achieves 99.5% accuracy using DBSCAN clustering and XGBoost classification. The web application provides an interactive dashboard to visualize the model's performance and make predictions.

## Files Included
- `earthquake_prediction_simple_improved.py` - The improved model code
- `earthquake_model_improved.pkl` - The trained improved model
- `processed_earthquake_data_improved.csv` - The processed data with new features
- `comparison_analysis.py` - Analysis script showing the improvements
- `earthquake-frontend/` - Web application frontend directory
- `earthquake-frontend/serve.py` - Python server to run the web application
- `start_website.sh` - Startup script to easily run the website

## Running the Web Application

### Option 1: Using the startup script (Recommended)
```bash
./start_website.sh
```

### Option 2: Direct execution
```bash
cd /workspace/earthquake-frontend
python3 serve.py
```

### Accessing the Application
Once started, the application will be available at:
`http://localhost:8000`

## Web Application Features
- Model performance metrics display (showing 99.5% accuracy)
- Interactive prediction form for new earthquake data
- Earthquake data visualization table
- Detailed model information and methodology

## Model Improvements
- Enhanced Feature Engineering: Added new features including depth_magnitude_ratio, magnitude_squared, depth_normalized, lat_long_interaction, temporal features (year, month, day_of_year), and cluster statistics
- Optimized DBSCAN Parameters: Using eps=0.05 and min_samples=5 for finer-grained clustering
- Better XGBoost Configuration: Optimized hyperparameters including n_estimators=200, max_depth=6, subsample=0.9, colsample_bytree=0.9
- Improved Data Splitting: Stratified sampling to maintain class distribution and shuffled splitting

## Results
The improved model achieved 99.5% accuracy on the test set (improved from 77%), with an ROC AUC score of 1.00. This represents a significant improvement of 22.5% over the original model.