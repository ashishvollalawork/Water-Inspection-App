# 💧 Water Inspection App

A professional water quality analysis platform built with **Streamlit** and **Machine Learning**.

## Features

✨ **ML-Powered Analysis** - Advanced machine learning algorithms for accurate water quality assessment
📊 **Instant Results** - Get real-time insights on water safety, quality, and risk levels
📋 **PDF Reports** - Generate professional A4 PDF reports with complete analysis data
🎯 **Brand Selection** - Track fixture brands (switchboards, faucets, wash basins, WC)
💾 **Auto-Load** - Automatically loads last inspection on app startup
🎨 **Modern UI** - Beautiful, responsive interface with smooth animations

## Parameters Analyzed

- **TDS (Total Dissolved Solids)** - Measures mineral content in water
- **pH Level** - Measures water acidity/alkalinity (safe range: 6.5-8.5)
- **Water Quality** - Classified as Good, Fair, Poor, or Very Poor
- **Drinking Status** - ML-based prediction: Safe or Not Safe
- **Risk Level** - Low, Medium, or High based on parameters

## Technology Stack

- **Frontend**: Streamlit with custom CSS animations
- **ML Model**: Scikit-learn (joblib-based artifact)
- **PDF Generation**: ReportLab
- **Typography**: Google Fonts Poppins
- **Data Handling**: Pandas, NumPy

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd WaterInspectionApp

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

```
WaterInspectionApp/
├── app.py                      # Main Streamlit application
├── artifacts/
│   ├── drinkstatus_model.pkl  # Trained ML model
│   └── drinkstatus_label_encoder.pkl
├── brand_master.csv            # Fixture brands database
├── last_inspection.json         # Auto-saved inspection data
├── .gitignore
└── README.md
```

## Workflow

1. **Water Testing** - Enter TDS and pH values, click Analyze
2. **Results** - View instant analysis with quality metrics and recommendations
3. **Brand Selection** - Select fixture brands (optional)
4. **Generate Report** - Create professional PDF or save inspection data

## Features Overview

### Water Analysis
- Real-time TDS and pH validation
- ML-based drinking water safety prediction
- Automatic risk assessment
- Detailed recommendations based on parameters

### Results Dashboard
- Color-coded status indicators (Safe/Warning/Danger)
- Quality metrics with animated visualizations
- Safety confidence percentage
- Breakdown of TDS and pH assessments

### Report Generation
- Professional A4 PDF format
- Complete water analysis data
- Fixture brand information
- Automatic timestamp and inspection details

## Design Highlights

🎨 **Modern Glassmorphism** - Frosted glass effects with backdrop blur
⚡ **Smooth Animations** - 60fps CSS animations with staggered effects
🎯 **Accessibility** - Respects `prefers-reduced-motion` preferences
📱 **Responsive** - Works seamlessly on desktop and tablet

## Author

Built with passion for clean water & clean code  
*25+ years of front-end engineering excellence*

---

**License**: MIT  
**Status**: Active Development
