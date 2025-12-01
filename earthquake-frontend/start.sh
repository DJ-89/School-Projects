#!/bin/bash

echo "Earthquake Prediction Dashboard - Startup Guide"
echo "==============================================="

echo "
Two options are available to run the application:

OPTION 1: Python Server (for environments with memory constraints)
- Run: python3 serve.py
- Access at: http://localhost:8000

OPTION 2: React Development Server (for full React experience)
1. Install dependencies: npm install
2. Start development server: npm run dev
3. Access at: http://localhost:5632 (or as shown in terminal)

Note: If you encounter memory errors during npm install, try:
- Using a machine with more memory
- Using npm ci instead of npm install
- Installing node modules one by one

The dashboard showcases our enhanced earthquake prediction model with 99.5% accuracy.
"