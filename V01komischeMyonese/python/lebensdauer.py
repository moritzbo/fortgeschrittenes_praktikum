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

Counts, Kanal = np.genfromtxt("data/new2.txt", unpack=True)

Countsextra = Counts
Countserr = np.sqrt(Countsextra)

Counts = Counts - Untergrundchannel

a = 0.0223
b = -0.0148

tliste = a*  Kanal + b

#for n in range(len(Counts) - m):
#    if unp.nominal_values(Counts[n]) <= 0:
#        Counts = np.delete(Counts, n)
#        tliste = np.delete(tliste, n)
#        Countserr = np.delete(Countserr, n)
#        #print(Counts[n])
#        #print(n)
#    elif n == len(Counts):
            #break
#print(unp.nominal_values(Counts))




def sigmoid1(x, A, B):
    return A * np.exp(- B*x)


params, covariance_matrix = curve_fit(sigmoid1, tliste, unp.nominal_values(Counts))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params:")
for name, value, uncertainty in zip('AB', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')



plt.errorbar(tliste, unp.nominal_values(Counts), xerr= 0, yerr = Countserr,  fmt='kx', elinewidth=0.7, label="Messdaten",markersize=3, capsize=1.5, markeredgewidth=0.5)

tyo = np.linspace(0, 10.6)
plt.plot(tyo, params[0] * np.exp(- params[1] * tyo), "b-", label="Ausgleichsfunktion")
plt.xlabel(r'$t$[$\si{\nano\second}$]')
plt.ylabel(r'$\text{N}$[$\text{Imp}$]')
plt.legend()
plt.grid()

lebensdauer = 1/ufloat(params[1], uncertainties[1])
print(lebensdauer)
plt.tight_layout()
plt.savefig("build/plotmessung.pdf")

plt.clf()

plt.errorbar(tliste, unp.nominal_values(Counts), xerr= 0, yerr = Countserr,  fmt='kx', elinewidth=0.7, label="Messdaten",markersize=3, capsize=1.5, markeredgewidth=0.5)

tyo = np.linspace(0, 10.5)
plt.plot(tyo, params[0] * np.exp(- params[1] * tyo), "b-", label="Ausgleichsfunktion")
plt.xlabel(r'$t$[$\si{\nano\second}$]')
plt.ylabel(r'$\text{N}$[$\text{Imp}$]')
plt.legend()
plt.grid()

plt.yscale("symlog")
plt.savefig("build/test3.pdf")

ttheo = 2.197

abweichung = 100 * (ttheo - lebensdauer)/ttheo

print(f"{abweichung:.2f}")