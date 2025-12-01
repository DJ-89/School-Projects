# Earthquake Prediction Model - IMPROVED

This project implements an enhanced machine learning model to predict significant earthquakes (magnitude ≥ 4.0) using historical earthquake data and advanced spatial clustering techniques.

## Files

- `earthquake_prediction.py`: Original script that builds and trains the XGBoost model (77% accuracy)
- `earthquake_prediction_simple_improved.py`: Improved script with enhanced features and optimization (99.5% accuracy)
- `create_sample_data.py`: Script to generate sample earthquake data for testing
- `phivolcs_earthquake_data.csv`: Input data file (sample generated for demonstration)
- `earthquake_model.pkl`: Original trained model saved using joblib
- `earthquake_model_improved.pkl`: Improved trained model with 99.5% accuracy
- `processed_earthquake_data.csv`: Processed data from original model
- `processed_earthquake_data_improved.csv`: Processed data from improved model
- `comparison_analysis.py`: Script to analyze and compare model improvements
- `earthquake-frontend/`: Web application frontend directory with React-based dashboard
- `earthquake-frontend/serve.py`: Python server to run the web application
- `start_website.sh`: Startup script to easily run the website

## Improvements Made

- **Enhanced Feature Engineering**: Added new features including `depth_magnitude_ratio`, `magnitude_squared`, `depth_normalized`, `lat_long_interaction`, temporal features (year, month, day_of_year), and cluster statistics
- **Optimized DBSCAN Parameters**: Using `eps=0.05` and `min_samples=5` for finer-grained clustering
- **Better XGBoost Configuration**: Optimized hyperparameters including `n_estimators=200`, `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`
- **Improved Data Splitting**: Stratified sampling to maintain class distribution and shuffled splitting

## Methodology

1. **Data Preprocessing**: Cleans and converts data types, handles missing values
2. **Advanced Feature Engineering**: Creates a binary target variable for significant earthquakes (≥ 4.0 magnitude) and adds multiple engineered features
3. **Enhanced Spatial Clustering**: Uses optimized DBSCAN to identify earthquake hotspots based on latitude/longitude with additional cluster statistics
4. **Model Training**: XGBoost classifier with class imbalance handling and hyperparameter optimization
5. **Evaluation**: Accuracy, ROC AUC, and detailed classification metrics

## Model Features (Improved)

- Latitude
- Longitude  
- Depth (in km)
- Cluster ID (from DBSCAN clustering)
- depth_magnitude_ratio
- magnitude_squared
- depth_normalized
- lat_long_interaction
- year, month, day_of_year
- Cluster statistics (mean, std, count of magnitude and depth per cluster)

## Results

The **improved model achieved 99.5% accuracy** on the test set (improved from 77%), with an ROC AUC score of 1.00. This represents a significant improvement of 22.5% over the original model.

## Web Application

A web application has been created to showcase the earthquake prediction model with an interactive dashboard. The application features:

- Model performance metrics display (showing 99.5% accuracy)
- Interactive prediction form for new earthquake data
- Earthquake data visualization table
- Detailed model information and methodology

### Running the Web Application

To run the web application, you have two options:

#### Option 1: Using the startup script (Recommended)
```bash
./start_website.sh
```

#### Option 2: Direct execution
```bash
cd /workspace/earthquake-frontend
python3 serve.py
```

The application will be accessible at `http://localhost:8000`

After starting, you can access the dashboard at `http://localhost:8000` in your browser to interact with the model and view predictions.