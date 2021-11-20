import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


# matplotlib.pyplot.rcdefaults()



deltat, kanal = np.genfromtxt("data/kanal.dat", unpack=True)


def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1, kanal, deltat)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.6f} ± {uncertainty:.6f}')

plt.plot(kanal, deltat, "kx", label="Messdaten", linewidth=1.5 )

x = np.linspace(0, 2900)
plt.plot(x, params[0]*x+params[1], 
        'b--',
        label="Lineare Ausgleichsgerade",
        linewidth=1)
plt.legend()
plt.grid()
plt.xlabel(r'Channel')
plt.ylabel(r'$\increment t$[$\si{\micro\second}$]')
plt.savefig("build/plot2.pdf")