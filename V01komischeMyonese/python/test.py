import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

t20, p20 = np.genfromtxt("../data/delay20.dat", unpack=True)
t10, p10 = np.genfromtxt("../data/delay10.dat", unpack=True)
            
t21, p21 = np.genfromtxt("../data/delay21.dat", unpack=True)
t22, p22 = np.genfromtxt("../data/delay22.dat", unpack=True)
                                         
t11, p11 = np.genfromtxt("../data/delay11.dat", unpack=True)
t12, p12 = np.genfromtxt("../data/delay12.dat", unpack=True)


plt.plot(t20, p20,
            'kx',
            label="Messdaten bei einer Breite von 20",
            linewidth=1.5)



plt.plot(t10, p10,
            'bx',
            label="Messdaten bei einer Breite von 10",
            linewidth=1.5)


def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1, t21, p21)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-22,0)

plt.plot(x, params[0]*x+params[1], 
        'r--',
        label="lineare Regression",
        linewidth=1)


def sigmoid2(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid2, t22, p22)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Abstieg:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0, 22)

plt.plot(x, params[0]*x+params[1], 
        'r--',
        linewidth=1)

def sigmoid3(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid3, t11, p11)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg 10er breite:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-13,0)

plt.plot(x, params[0]*x+params[1], 
        'r--',
        linewidth=1)

def sigmoid4(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid4, t12, p12)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg 10er breite:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0, 12)

plt.plot(x, params[0]*x+params[1], 
        'r--',
        linewidth=1)

# plt.ylabel(r'Impulse $1/10 [\si{\per\seconds}]$')
# plt.xlabel(r'$\increment t$ [\si{\seconds}]$')


# plt.legend(loc="upper left")
# plt.grid()
plt.savefig("../build/plot1.pdf")
