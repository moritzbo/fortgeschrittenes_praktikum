import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

dB, mm, U, K= np.genfromtxt("Data/eich.dat", unpack=True)

plt.plot(mm, dB, 'bx',
        label='Hersteller Angabe'
        )
plt.plot(U, dB, "gx",
        label="Messwerte"
        )
plt.plot(U, K, "ro",
        label="Korrigierte Messwerte"
        )        

def sigmoid1(x, a, b, c):
    return a*x**2 + b*x + c


params, covariance_matrix = curve_fit(sigmoid1, mm, dB)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("MODE 1:")
for name, value, uncertainty in zip('abc', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')



x = np.linspace(0,3)

plt.plot(x, 
        params[0]*x**2 + params[1]*x +params[2],
        'k-',
        label='Eichkurve',
        linewidth=1.5)



plt.ylabel(r'$\si{\decibel}$')
plt.xlabel(r'$d$[$\si{\milli\meter}$]$')


plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("build/plot3.pdf")
# plt.show()



