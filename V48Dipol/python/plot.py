import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


# matplotlib.pyplot.rcdefaults()


t1a, T1a, i1a = np.genfromtxt("../data/A1TproM.dat", unpack=True)
t1p, T1p, i1p = np.genfromtxt("../data/Peaks1TproM.dat", unpack=True)


t2a, T2a, i2a = np.genfromtxt("../data/A2TproM.dat", unpack=True)
t2p, T2p, i2p = np.genfromtxt("../data/Peaks2TproM.dat", unpack=True)


plt.plot(T1a, i1a, "xb")
plt.plot(T1p, i1p, "xr")

def sigmoid1(x, a, b):
    return a*np.exp(b*x)


params1, covariance_matrix = curve_fit(sigmoid1, T1a, i1a)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg:")
for name, value, uncertainty in zip('ab', params1, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-70,40)

plt.plot(x, params1[0]*np.exp(params1[1]*x),
        'k--',
        label="lineare Regression",
        linewidth=1)


plt.plot(T2a, i2a, "xb")
plt.plot(T2p, i2p, "xr")

def sigmoid1(x, a, b):
    return a*np.exp(b*x)


params2, covariance_matrix = curve_fit(sigmoid1, T2a, i2a)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg:")
for name, value, uncertainty in zip('ab', params2, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-70,40)

plt.plot(x, params2[0]*np.exp(params2[1]*x),
        'k--',
        label="lineare Regression",
        linewidth=1)


plt.show()
plt.clf()

plt.plot(T1a, i1a-(params1[0]*np.exp(params1[1]*T1a)), "xb")
plt.plot(T1p, i1p-(params1[0]*np.exp(params1[1]*T1p)), "xr")
plt.xlim(-60,40)
plt.ylim(-1,2)
plt.show()

plt.clf()

plt.plot(T2a, i2a-(params2[0]*np.exp(params2[1]*T2a)), "xb")
plt.plot(T2p, i2p-(params2[0]*np.exp(params2[1]*T2p)), "xr")
plt.xlim(-60,50)
plt.ylim(-1,2)
plt.show()