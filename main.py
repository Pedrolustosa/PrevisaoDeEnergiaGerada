from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

app = FastAPI(title="Previsão de Energia Gerada - Todos os Modelos")

# Adicione a configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 
                   "https://poli-previsao-energia-solar.vercel.app", 
                   "http://poli-previsao-energia-solar.vercel.app"], # Ou ["*"] para liberar tudo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup R forecast ---
ro.r('library(forecast)')
arima = ro.r('auto.arima')
arimaTest = ro.r('Arima')
ordem = ro.r('cbind')
fitted = ro.r('fitted')

# Variáveis globais (para todos os modelos)
test_dates = None

# ARIMA
arima_test = None
predArimaTest = None

# ARIMAX
arimax_test = None
predArimaxTest = None

# SVR
predTestSVR = None
testTargetSVR_dnorm = None

# MLP
predTestMLP = None
testTargetMLP_dnorm = None

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global test_dates, arima_test, predArimaTest, arimax_test, predArimaxTest
    global predTestSVR, testTargetSVR_dnorm, predTestMLP, testTargetMLP_dnorm

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Envie um arquivo CSV.")

    df = pd.read_csv(file.file)
    # Coluna 0 = data, 1 = target, 2,3,4 = exógenas
    dates = df.iloc[:, 0].astype(str).values
    data = df.iloc[:, 1].values
    n = len(data)
    train_size = int(np.floor(0.6 * n))
    val_size = int(np.floor(0.2 * n))
    test_idx = train_size + val_size

    # --------- ARIMA -------------
    arima_train = data[:train_size]
    arima_val = data[train_size:test_idx]
    arima_test = data[test_idx:]
    test_dates = dates[test_idx:]

    mdl_arima = arima(np.array(arima_train))
    fitTesteArima = arimaTest(np.array(arima_test), model=mdl_arima)
    predArimaTest = np.array(fitted(fitTesteArima))

    # --------- ARIMAX -------------
    # Só prepara se houver ao menos 3 exógenas
    if df.shape[1] >= 5:
        exog1 = df.iloc[:, 2].values
        exog2 = df.iloc[:, 3].values
        exog3 = df.iloc[:, 4].values

        arimax_train = data[:train_size]
        arimax_val = data[train_size:test_idx]
        arimax_test = data[test_idx:]

        xreg_train = ordem(
            ro.FloatVector(exog1[:train_size]),
            ro.FloatVector(exog2[:train_size]),
            ro.FloatVector(exog3[:train_size])
        )
        xreg_val = ordem(
            ro.FloatVector(exog1[train_size:test_idx]),
            ro.FloatVector(exog2[train_size:test_idx]),
            ro.FloatVector(exog3[train_size:test_idx])
        )
        xreg_test = ordem(
            ro.FloatVector(exog1[test_idx:]),
            ro.FloatVector(exog2[test_idx:]),
            ro.FloatVector(exog3[test_idx:])
        )

        mdl_arimax = arima(np.array(arimax_train), xreg=xreg_train)
        fit_Arimax_Test = arimaTest(np.array(arimax_test), xreg=xreg_test, model=mdl_arimax)
        predArimaxTest = np.array(fitted(fit_Arimax_Test))
        arimax_test = arimax_test
    else:
        predArimaxTest = None
        arimax_test = None

    # --------- SVR -------------
    serie = data
    maxData = np.max(serie)
    minData = np.min(serie)
    ndataset = (serie - minData) / (maxData - minData)
    datasetSeries = pd.Series(ndataset)
    dimension = 2
    stepahead = 1

    datasetShifted = pd.concat([datasetSeries.shift(i) for i in range(dimension + stepahead)], axis=1)
    train = datasetShifted.iloc[dimension + stepahead - 1:train_size, 1:]
    trainTarget = datasetShifted.iloc[dimension + stepahead - 1:train_size, 0]
    valid = datasetShifted.iloc[train_size:test_idx, 1:]
    validTarget = datasetShifted.iloc[train_size:test_idx, 0]
    test = datasetShifted.iloc[test_idx:, 1:]
    testTarget = datasetShifted.iloc[test_idx:, 0]

    g = (10.0) ** np.arange(-5, 3, 1)
    e = (10.0) ** np.arange(-4, -1, 1)
    c = (10.0) ** np.arange(-2, 3, 1)
    bestValue = 1e20
    bestSVR = None
    for i in g:
        for j in e:
            for k in c:
                mySVR = SVR(C=k, gamma=i, epsilon=j)
                mySVR.fit(train, trainTarget)
                predVals = mySVR.predict(valid)
                erro = mean_squared_error(predVals, validTarget)
                if erro < bestValue:
                    bestValue = erro
                    bestSVR = mySVR
    predTestSVR = bestSVR.predict(test) * (maxData - minData) + minData
    testTargetSVR_dnorm = testTarget * (maxData - minData) + minData

    # --------- MLP -------------
    maxDataMLP = np.max(serie[:train_size])
    minDataMLP = np.min(serie[:train_size])
    ndatasetMLP = (serie - minDataMLP) / (maxDataMLP - minDataMLP)
    datasetSeriesMLP = pd.Series(ndatasetMLP)
    dimensionMLP = 3
    stepaheadMLP = 1

    datasetShiftedMLP = pd.concat([datasetSeriesMLP.shift(i) for i in range(dimensionMLP + stepaheadMLP)], axis=1)
    trainMLP = datasetShiftedMLP.iloc[dimensionMLP + stepaheadMLP - 1:train_size, stepaheadMLP:]
    trainTargetMLP = datasetShiftedMLP.iloc[dimensionMLP + stepaheadMLP - 1:train_size, 0]
    validMLP = datasetShiftedMLP.iloc[train_size:test_idx, stepaheadMLP:]
    validTargetMLP = datasetShiftedMLP.iloc[train_size:test_idx, 0]
    testMLP = datasetShiftedMLP.iloc[test_idx:, stepaheadMLP:]
    testTargetMLP = datasetShiftedMLP.iloc[test_idx:, 0]

    hidden_neurons = [(10,), (20,), (30,), (40,), (50,)]
    bestValueMLP = np.inf
    bestMLP = None
    for i in hidden_neurons:
        myMLP = MLPRegressor(hidden_layer_sizes=i, max_iter=500)
        myMLP.fit(trainMLP, trainTargetMLP)
        predVals = myMLP.predict(validMLP)
        erro = mean_squared_error(predVals, validTargetMLP)
        if erro < bestValueMLP:
            bestValueMLP = erro
            bestMLP = myMLP
    predTestMLP = bestMLP.predict(testMLP) * (maxDataMLP - minDataMLP) + minDataMLP
    testTargetMLP_dnorm = testTargetMLP * (maxDataMLP - minDataMLP) + minDataMLP

    return {
        "message": "Modelos treinados e previsões dos blocos de teste calculadas.",
        "test_dates": test_dates.tolist()
    }

@app.get("/predict/arima")
def predict_arima():
    global arima_test, predArimaTest, test_dates
    if predArimaTest is None or arima_test is None:
        raise HTTPException(status_code=400, detail="Primeiro faça upload do CSV.")
    result = []
    for dt, real, pred in zip(test_dates, arima_test, predArimaTest):
        result.append({"date": dt, "real": float(real), "predicted": float(pred)})
    return result

@app.get("/predict/arimax")
def predict_arimax():
    global arimax_test, predArimaxTest, test_dates
    if predArimaxTest is None or arimax_test is None:
        raise HTTPException(status_code=400, detail="Primeiro faça upload do CSV ou verifique se há exógenas suficientes.")
    result = []
    for dt, real, pred in zip(test_dates, arimax_test, predArimaxTest):
        result.append({"date": dt, "real": float(real), "predicted": float(pred)})
    return result

@app.get("/predict/svr")
def predict_svr():
    global predTestSVR, testTargetSVR_dnorm, test_dates
    if predTestSVR is None or testTargetSVR_dnorm is None:
        raise HTTPException(status_code=400, detail="Primeiro faça upload do CSV.")
    result = []
    for dt, real, pred in zip(test_dates, testTargetSVR_dnorm, predTestSVR):
        result.append({"date": dt, "real": float(real), "predicted": float(pred)})
    return result

@app.get("/predict/mlp")
def predict_mlp():
    global predTestMLP, testTargetMLP_dnorm, test_dates
    if predTestMLP is None or testTargetMLP_dnorm is None:
        raise HTTPException(status_code=400, detail="Primeiro faça upload do CSV.")
    result = []
    for dt, real, pred in zip(test_dates, testTargetMLP_dnorm, predTestMLP):
        result.append({"date": dt, "real": float(real), "predicted": float(pred)})
    return result