import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

#zunächst komische Rechnungen

Nges = ufloat(3256768, np.sqrt(3256768))
Tges = 272190
Tsuch = 10**(-5)
Rate = Nges/Tges

Nsuch = Rate * Tsuch

warsch = Nsuch * unp.exp(-Nsuch)

Untergrundges = Nges * warsch
Untergrundchannel = Untergrundges/511

# print(Rate)
# print(Nsuch)
# print(warsch)
# print(Untergrundges)
# print(Untergrundchannel)

Counts, Kanal = np.genfromtxt("../data/new.txt", unpack=True)

Countsextra = Counts
Counts = Counts - Untergrundchannel
for n in range(511):
    if unp.nominal_values(Counts[n]) < 0:
        Counts[n] = 0

a = 0.0223
b = -0.0148

tliste = a*  Kanal + b


def sigmoid1(x, A, B, C):
    return A * np.exp(- B*x) + C


params, covariance_matrix = curve_fit(sigmoid1, tliste, unp.nominal_values(Counts))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params:")
for name, value, uncertainty in zip('ABC', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

Countserr = np.sqrt(Countsextra)

plt.errorbar(tliste, Counts, xerr= 0, yerr = Countserr,  fmt='kx', elinewidth=0.7, label="Messdaten",markersize=3, capsize=1.5, markeredgewidth=0.5)

plt.show()
