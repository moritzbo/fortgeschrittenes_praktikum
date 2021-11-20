import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


# matplotlib.pyplot.rcdefaults()


t20, p20 = np.genfromtxt("data/delay20.dat", unpack=True)
t10, p10 = np.genfromtxt("data/delay10.dat", unpack=True)
            
t21, p21 = np.genfromtxt("data/delay21.dat", unpack=True)
t22, p22 = np.genfromtxt("data/delay22.dat", unpack=True)
                                
t11, p11 = np.genfromtxt("data/delay11.dat", unpack=True)
t12, p12 = np.genfromtxt("data/delay12.dat", unpack=True)


p20err = np.sqrt(p20)
p10err = np.sqrt(p10)

#plt.plot(t20, p20,
#            'kx',
#            label="Messdaten mit Fehlerbalken bei einer Breite von 20",
#            linewidth=1.5)
plt.errorbar(t20, p20, xerr=None, yerr=p20err, fmt='kx', elinewidth=1, label="Messdaten bei Breite 20") 


# plt.plot(t10, p10,
#             'bx',
#             label="Messdaten bei einer Breite von 10",
#             linewidth=1.5)


def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1, t21, p21)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-22,0)

plt.plot(x, params[0]*x+params[1], 
        'b--',
        label="Ausgleichgerade Links",
        linewidth=1)
paras1 = ufloat(params[1], uncertainties[1])
paras0 = ufloat(params[0], uncertainties[0])

tlinks20 = (127 - paras1)/paras0



def sigmoid2(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid2, t22, p22)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Abstieg:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0, 22)

plt.plot(x, params[0]*x+params[1], 
        'c--',
        label="Ausgleichgerade Rechts",
        linewidth=1)

paras1 = ufloat(params[1], uncertainties[1])
paras0 = ufloat(params[0], uncertainties[0])
trechts20 = (127 - paras1)/paras0

plt.ylim(-20)
plt.vlines(unp.nominal_values(tlinks20), -20, 127, colors='r', linestyle='dashed')
plt.vlines(unp.nominal_values(trechts20), -20, 127, colors='r', linestyle='dashed')
plt.hlines(127, unp.nominal_values(tlinks20), unp.nominal_values(trechts20), colors='r', linestyle='dashed', label='Halbwertsbreite')
plt.legend(prop={'size': 8})
plt.grid()
plt.xlabel(r'$t$[$\si{\nano\second}$]')
plt.ylabel(r'$\text{N}$[$\text{Imp}\si{\per{10}\second}$]')
plt.savefig("build/plot11.pdf")
plt.clf()

#t[\si{\nano\second}]

def sigmoid3(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid3, t11, p11)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg 10er breite:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(-13,0)

plt.plot(x, params[0]*x+params[1], 
        'b--',
        label="Ausgleichgerade Links",
        linewidth=1)
paras1 = ufloat(params[1], uncertainties[1])
paras0 = ufloat(params[0], uncertainties[0])
tlinks10 = (67 - paras1)/paras0

def sigmoid4(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid4, t12, p12)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg 10er breite:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0, 12)

plt.plot(x, params[0]*x+params[1], 
        'c--',
        label="Ausgleichgerade Rechts",
        linewidth=1)
paras1 = ufloat(params[1], uncertainties[1])
paras0 = ufloat(params[0], uncertainties[0])
trechts10 = (67 - paras1)/paras0

#plt.ylabel(r'Impulse $1/10 [\si{\per\seconds}]$')
# # plt.xlabel(r'$\increment t$ [\si{\seconds}]$')


# # plt.legend(loc="upper left")
# # plt.grid()

# print(f"Tlinks20: {tlinks20:.2f}")
# print(f"Trechts20: {trechts20:.2f}")
# print(f"Tlinks10: {tlinks10:.2f}")
# print(f"Trechts10: {trechts10:.2f}")

# T20 = trechts20 - tlinks20
# T10 = trechts10 - tlinks10

# print(f"T20: {T20:.2f}")
# print(f"T10: {T10:.2f}")

plt.errorbar(t10, p10, xerr=None, yerr=p10err, fmt='kx', elinewidth=1, label="Messdaten bei Breite 10") 
plt.ylim(-10)
plt.vlines(unp.nominal_values(tlinks10), -10, 67, colors='r', linestyle='dashed')
plt.vlines(unp.nominal_values(trechts10), -10, 67, colors='r', linestyle='dashed')
plt.hlines(67, unp.nominal_values(tlinks10), unp.nominal_values(trechts10), colors='r', linestyle='dashed', label='Halbwertsbreite')
plt.legend(prop={'size': 8})
plt.grid()
plt.xlabel(r'$t$[$\si{\nano\second}$]')
plt.ylabel(r'$\text{N}$[$\text{Imp}\si{\per{10}\second}$]')

plt.savefig("build/plot12.pdf")

# plt.hlines(127, unp.nominal_values(tlinks20), unp.nominal_values(trechts20), colors=None, linestyle='solid', label='Halbwertsbreite ')
# plt.hlines(67, unp.nominal_values(tlinks10), unp.nominal_values(trechts10), colors=None, linestyle='solid', label='Halbwertsbreite ')




