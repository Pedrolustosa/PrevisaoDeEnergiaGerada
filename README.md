# Energy Generation Prediction API

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![R](https://img.shields.io/badge/r-%23276DC3.svg?style=flat&logo=r&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive FastAPI-based machine learning service for energy generation forecasting using multiple time series prediction models including ARIMA, ARIMAX, Support Vector Regression (SVR), and Multi-Layer Perceptron (MLP).

## 🎯 Project Overview

This project provides a robust API for predicting energy generation using historical time series data. The system implements and compares four different machine learning approaches:

- **ARIMA**: AutoRegressive Integrated Moving Average for univariate time series forecasting
- **ARIMAX**: ARIMA with eXogenous variables for multivariate time series forecasting
- **SVR**: Support Vector Regression with hyperparameter optimization
- **MLP**: Multi-Layer Perceptron neural network with architecture optimization

The API automatically splits data into training (60%), validation (20%), and testing (20%) sets, trains all models, and provides predictions for comparison and evaluation.

## 🚀 Features

- **Multi-Model Approach**: Compare predictions from four different algorithms
- **Automatic Hyperparameter Tuning**: Grid search optimization for SVR and MLP models
- **CORS Support**: Ready for web application integration
- **Docker Support**: Containerized deployment with R and Python dependencies
- **RESTful API**: Clean endpoints for data upload and model predictions
- **Data Validation**: Automatic CSV format validation and error handling

## 📋 Prerequisites

### Requirements
- Python 3.9+
- R with `forecast` package
- Docker (optional)

### Dependencies
```
fastapi
uvicorn
pandas
numpy
scikit-learn
statsmodels
python-multipart
rpy2==3.5.0
```

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd PrevisaoDeEnergiaGerada
```

2. **Install R and required packages**
```bash
sudo apt-get update
sudo apt-get install r-base r-base-dev
R -e "install.packages('forecast', repos='http://cran.us.r-project.org')"
```

3. **Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. **Build the Docker image**
```bash
docker build -t energy-prediction-api .
```

2. **Run the container**
```bash
docker run -p 8000:8000 energy-prediction-api
```

## 🚀 Running the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000`

## 🔌 API Endpoints

- `POST /upload_csv` - Upload CSV file and train all models
- `GET /predict/arima` - Get ARIMA model predictions
- `GET /predict/arimax` - Get ARIMAX model predictions (requires exogenous variables)
- `GET /predict/svr` - Get Support Vector Regression predictions
- `GET /predict/mlp` - Get Multi-Layer Perceptron predictions

## 📈 Model Details

### ARIMA/ARIMAX
- Uses R's `auto.arima()` function for automatic parameter selection
- ARIMAX requires at least 5 columns in CSV (date + target + 3 exogenous variables)
- Optimal parameters (p,d,q) are automatically determined

### Support Vector Regression (SVR)
- **Window size**: 2 time steps
- **Hyperparameter grid search**:
  - Gamma: [0.0001, 0.001, 0.01]
  - Epsilon: [0.0001, 0.001, 0.01]
  - C: [0.01, 0.1, 1, 10, 100]
- Data normalization applied

### Multi-Layer Perceptron (MLP)
- **Window size**: 3 time steps
- **Architecture search**: [10, 20, 30, 40, 50] hidden neurons
- **Training**: 500 maximum iterations
- Data normalization applied
## 📝 Usage Example

```python
import requests

with open('energy_data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload_csv', files={'file': f})

arima_predictions = requests.get('http://localhost:8000/predict/arima').json()
svr_predictions = requests.get('http://localhost:8000/predict/svr').json()
```

## 👨‍💻 Author

**Pedro Lustosa**  
LinkedIn: [https://www.linkedin.com/in/pedrolustosadev/](https://www.linkedin.com/in/pedrolustosadev/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions, please open an issue in the repository or contact the development team.

---

**Note**: This API is designed for research and educational purposes.
