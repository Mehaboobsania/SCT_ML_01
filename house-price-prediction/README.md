# House Price Prediction

This project implements an end-to-end Machine Learning pipeline using Scikit-Learn to predict house prices.

## Project Structure
```
house-price-prediction/
├── data/                      # Folder for datasets (train.csv, test.csv)
├── src/                       # Source files containing modular logic
│   ├── __init__.py            
│   ├── utils.py               
│   ├── eda.py                 
│   ├── preprocessing.py       
│   └── model.py               
├── models/                    # Serialized models
├── outputs/                   # Generated plots and predictions
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── app.py                     # Streamlit web dashboard
└── main.py                    # Main pipeline orchestrator script
```

## Running the Project

### Setup
Ensure the virtual environment is activated:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (Command Prompt): `.venv\Scripts\activate.bat`

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Option 1: Main Pipeline
Runs training, diagnostics, and test inference:
```bash
python main.py
```

### Option 2: Streamlit Web Dashboard
Launches the interactive dashboard:
```bash
streamlit run app.py
```
This is perfect for recording real-time predictions and displaying graphs.
