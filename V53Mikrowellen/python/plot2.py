import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

v1, m1, v2, m2, v3, m3= np.genfromtxt("Data/mode2.txt", unpack=True)

plt.plot(v1, m1,
            'ko',
            label="Messdaten",
            linewidth=1.5)
plt.plot(v2, m2,
            'ko',
            linewidth=1.5)
plt.plot(v3, m3,
            'ko',
            linewidth=1.5)



#### MODE - miliVOLT
def sigmoid1(x, a, b, c):
    return a*x**2+b*x+c
print("MODE FÜR FREQUENZZZZZZZZZZZZZ")

params, covariance_matrix = curve_fit(sigmoid1, v1, m1)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("MODE 1")
for name, value, uncertainty in zip('abc', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')
print(params[0])
print(params[1])
print(params[2])

x = np.linspace(200,220)
plt.plot(x, 
        params[0]*x**2 + params[1]*x +params[2],
        'c-',
        label='Mode 1',
        linewidth=1.5)


def sigmoid2(x, a, b, c):
    return a*x**2+b*x+c

params, covariance_matrix = curve_fit(sigmoid2, v2, m2)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("MODE 2")
for name, value, uncertainty in zip('abc', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')
print(params[0])
print(params[1])
print(params[2])
x = np.linspace(115,152)
plt.plot(x, 
        params[0]*x**2 + params[1]*x +params[2],
        'b-',
        label='Mode 2',
        linewidth=1.5)

 
def sigmoid3(x, a, b, c):
    return a*x**2+b*x+c


print("MODE 3")
params, covariance_matrix = curve_fit(sigmoid3, v3, m3)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('abc', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')
print(params[0])
print(params[1])
print(params[2])
x = np.linspace(60,82)
plt.plot(x, 
        params[0]*x**2 + params[1]*x +params[2],
        'g-',
        label='Mode 3',
        linewidth=1.5)

plt.ylabel(r'$\increment f$[$\si{\mega\hertz}$]$')
plt.xlabel(r'$U_{\text{ref}}$[$\si{\volt}$]$')

plt.ylim((-60,60))
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("build/plot2.pdf")
plt.show()

