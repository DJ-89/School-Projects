#!/bin/bash
# Script to start the earthquake prediction web application

echo "Starting Earthquake Prediction Web Application..."
echo "The application will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

cd /workspace/earthquake-frontend
python3 serve.py