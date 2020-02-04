""" 
Universidade Federal de Pernambuco
Departamento de Enegia Nuclear
Soil Physics Laboratory
Author: Lucas Ravellys <ravellyspyrrho@gmail.com>

Routine to estime the soil water content by GaussianProcessRegressor in Caatinga area.          
We train the model with data  of the soil moisture and the 
precipitation cumulatede in 7 days (P7).
Thereafter, the data is simulated with the initial moisture (tho) and P7.

The data used in this work was provided by the project INCT-ONDACBC 
(Observatório Nacional da Dinâmica da Água e de Carbono no Bioma Caatinga)
The soil moisture was evalueted by TDR sensors in depth of 10,20,30, and 40 cm.
	
This tower is located in a seasonal tropical dry forest (Caatinga) 
in the semi-arid region of Brazil (Serra Talhada - PE). 
(http://dx.doi.org/10.17190/AMF/1562386)

Variables
dias = day
P7 = Precipitation cumulated in seven days
ETo = potential Evapotranpiration
tho = initial soil moisture
x_ = features
y_ = Target values
TH = soil moisture estimated

"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#import mensured data
dados_medidos = pd.read_csv("Dados_medidos .csv", header = 0, sep = ";")
dias = dados_medidos["dia"].values[6:]
P7= dados_medidos["P7"].values
ETo = dados_medidos["Eto"].values

#initial soil moistures
tho = dados_medidos[["th1","th2","th3","th4"]][5:6].values

# create features matrix. Note that features matrix has soil moisture data in time (i-1),
# and the P7 data in time (i) 
x_ = dados_medidos[["th1","th2","th3","th4"]][5:-1]
x_["P7"]= dados_medidos[["P7"]][6:].values
x_=np.atleast_2d(x_.values)


#create the Target values with soil moisture data in time (i) 
y_ = np.atleast_2d(dados_medidos[["th1","th2","th3","th4"]].values[6:])

# Instantiate a Gaussian Process model
kernel = C(1.,(1e-3,1e-3))*RBF(.1,(1e-2,1e-2))
gp = GaussianProcessRegressor(alpha=1e-7, copy_X_train=True,
                         kernel=kernel,
                         n_restarts_optimizer=99, normalize_y=True,
                         optimizer='fmin_l_bfgs_b')

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_, y_)

# Make the prediction
# In this loop the values are estimated with the initial moisture with interactive method
# were Th[i+1] is calculeted with TH[i]
TH = tho.tolist()
for i in range(len(y_)-1):
    print(TH[i])
    TH.append(gp.predict(np.atleast_2d(np.concatenate((TH[i],P7[i+6]), axis=None)))[0])
TH= np.array(TH)    
    
# Plot the mensured data and simulated data
for i in range(len(TH[0])):
    plt.plot(dados_medidos[["dia"]].values, dados_medidos[["th"+str(i+1)]].values, 'r.', markersize=5, label='Observations')
    plt.plot(dias, TH[:,i], 'b:', label=r'$Simulado$ th'+str(i+1))
    plt.xlabel('$dias$')
    plt.ylabel('$Umidade$')
    plt.ylim(0.02, .2)
    plt.legend(loc='upper right')
    plt.show()