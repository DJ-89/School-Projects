# Earthquake Prediction Dashboard

This is a React-based web application that visualizes the enhanced earthquake prediction model using DBSCAN and XGBoost with 99.5% accuracy.

## Features

- Real-time earthquake prediction based on latitude, longitude, and depth
- Visualization of recent earthquake data
- Model performance metrics display
- Detailed information about the methodology used

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn

## Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:

```bash
npm install
```

## Running the Application

To start the development server:

```bash
npm run dev
```

The application will be accessible at `http://localhost:5632` (or another available port as indicated in the terminal).

## Model Information

This dashboard showcases an enhanced earthquake prediction model that achieves 99.5% accuracy using:
- XGBoost classifier with optimized hyperparameters
- DBSCAN clustering for identifying seismic hotspots
- Enhanced feature engineering including depth-magnitude ratios and spatial interactions

The model predicts whether an earthquake will be significant (magnitude ≥ 4.0).
