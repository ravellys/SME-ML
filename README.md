# Estimating Soil Moisture and Actual Evapotranspiration with Machine Learning

Routine to estime the soil water content by GaussianProcessRegressor in Caatinga area. We train the model with data of the soil moisture, the precipitation, and the potencial evapotranspiration. The data is simulated with the initial conditions of the soil moisture, and the contour conditions of rainfall and potential evapotranspiration. The data used in this work was provided by the project INCT-ONDACBC (Observatório Nacional da Dinâmica da Água e de Carbono no Bioma Caatinga).The soil moisture was evalueted by TDR sensors in depth of 10,20,30, and 40 cm. This tower is located in a seasonal tropical dry forest (Caatinga) in the semi-arid region of Brazil (Serra Talhada - PE) (http://dx.doi.org/10.17190/AMF/1562386).


## Variables:
1. dias = day
2. cP = Precipitation
3. ETo = potential Evapotranpiration
4. ETa = actual Evapotranspiration
5. tho = initial soil moisture
6. x_ = features
7. y_ = Target values
8. TH = soil moisture estimated

## Initialy, we are import the follows packages:
- import numpy as np
- from matplotlib import pyplot as plt
- import seaborn as sns
- import pandas as pd
- from sklearn.gaussian_process import GaussianProcessRegressor
- from sklearn.gaussian_process.kernels import RBF, WhiteKernel
- from sklearn.model_selection import train_test_split
- import hydroeval as he

### import mensured data
dados_medidos = pd.read_csv("Dados_medidos .csv", header = 0, sep = ";")
dias = dados_medidos["dia"].values[init:]
cP= dados_medidos[cumulated_P].values
ETo = dados_medidos["Eto"].values


