# Flex-Settler: Demand Response Baseline Calculator

Flex-Settler is an interactive Streamlit application for calculating demand response baselines using industry-validated methods such as High X of Y, Middle 8 of 10, and same-day adjustments.  
It provides detailed analytics, gaming-risk detection, visualizations, and exportable reports.

This tool was developed as part of an energy analytics project.

---

## Key Features

- Multiple baseline methodologies:
  - High 5 of 10  
  - High 4 of 5  
  - High 10 of 15  
  - Middle 8 of 10  

- Same-day adjustment (additive or scalar) with configurable caps  
- Automatic similar-day detection (weekday/weekend logic)  
- 15-minute aligned event interval construction  
- Delivered energy calculation  
- Gaming risk analysis  
- Interactive visualizations using Plotly  
- Export to Excel and CSV  
- Supports both CSV and Excel data uploads  

---

## Project Structure

flex-settler/  
├── final.py  
├── requirements.txt  
└── README.md  

---

## Installation

### 1. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows


### 2. Install dependencies



pip install -r requirements.txt


---

## Running the Application


streamlit run final.py


The application will open at:

http://localhost:8501

---

## How It Works

### 1. Upload your data
Accepted formats:
- CSV
- XLS / XLSX

Required columns:
- Datetime  
- Consumption (kWh)

Optional:
- Activation column

### 2. Define the event period
- Select the date  
- Choose start and end times  

### 3. Configure methodology
- Choose High X of Y or Middle 8 of 10  
- Select adjustment type (additive or scalar)  
- Define adjustment cap  
- Select adjustment window duration  

### 4. View results
The app displays:
- Raw and adjusted baseline  
- Actual consumption  
- Delivered energy  
- Adjustment values  
- Gaming risk score and factors  
- Interactive plots  

### 5. Export results
- Excel (summary, detailed results, risk analysis)  
- CSV  

---

## Screenshots

![Baseline vs Actual Consumption and Gaming Risk Analysis – example 1](docs/Baseline vs Actual Consumption and Gaming Risk Analysis – example 1.png)
![Baseline vs Actual Consumption and Gaming Risk Analysis – example 2 with delivered energy area](Baseline vs Actual Consumption and Gaming Risk Analysis – example 2 with delivered energy area.png)






---

## Requirements



numpy
pandas
scikit-learn
plotly
streamlit
openpyxl


---

## Author

Kareem Hussein  
GitHub: https://github.com/kemogamd  
Email: khaledokareem@gmail.com
