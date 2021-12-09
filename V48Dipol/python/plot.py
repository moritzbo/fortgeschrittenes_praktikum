import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


# matplotlib.pyplot.rcdefaults()


t1a, T1ac, i1a = np.genfromtxt("../data/A1TproM.dat", unpack=True)
t1p, T1pc, i1p = np.genfromtxt("../data/Peaks1TproM.dat", unpack=True)
T1a= T1ac +273.15
print(T1a)
print(i1a)
T1p=T1pc + 273.15

t2a, T2ac, i2a = np.genfromtxt("../data/A2TproM.dat", unpack=True)
t2p, T2pc, i2p = np.genfromtxt("../data/Peaks2TproM.dat", unpack=True)
T2a= T2ac + 273.15
T2p= T2pc + 273.15

t1an, T1anc, i1an = np.genfromtxt("../data/anlauf1TproM.dat", unpack=True)
t2an, T2anc, i2an = np.genfromtxt("../data/anlauf2TproM.dat", unpack=True)
T1an= T1anc + 273.15
T2an= T2anc + 273.15

t1, Tneu1, dsfds = np.genfromtxt("../data/1GRADproMIN.dat", unpack=True)
t2, Tneu2, fsddf = np.genfromtxt("../data/2GRADproMIN.dat", unpack=True)
T1= Tneu1 + 273.15
T2= Tneu2 + 273.15

plt.plot(t1, T1, "ob", markersize="1.5", label="Messdaten")
plt.plot(t2, T2, "ob", markersize="1.5", )

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1, t1, T1)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Temperatur 1 grad pro mmin:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,120)

plt.plot(x, params[0]*x+params[1],
        'k--',
        
        linewidth=1)

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1, t2, T2)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Temperatur 2 grad pro mmin:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,60)

plt.plot(x, params[0]*x+params[1],
        'k--',
        label="lineare Regression",
        linewidth=1)    


plt.ylabel(r'T $[\si{\celsius}]$')
plt.xlabel(r't $[\si{\minutes}]$')

plt.grid()
plt.legend()
# plt.savefig("build/Temperatur.pdf")
plt.clf()




plt.plot(T1a, i1a, "xb")
plt.plot(T1p, i1p, "xr")

def sigmoid1(x, a, b):
    return a*np.exp(b*x)


params1, covariance_matrix = curve_fit(sigmoid1, T1a, i1a,p0=(1,0.01))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Untergrund1:")
for name, value, uncertainty in zip('ab', params1, uncertainties): 
    print(f'{name} = {value:.15f} ± {uncertainty:.15f}')

x = np.linspace(200,310)

plt.plot(x, params1[0]*np.exp(params1[1]*x),
        'k--',
        label="lineare Regression",
        linewidth=1)
plt.show()
plt.clf()
plt.plot(T2a, i2a, "xb")
plt.plot(T2p, i2p, "xr")

def sigmoid1(x, a, b):
    return a*np.exp(b*x)


params2, covariance_matrix = curve_fit(sigmoid1, T2a, i2a, p0=(1,0.01))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Untergrund2:")
for name, value, uncertainty in zip('ab', params2, uncertainties): 
    print(f'{name} = {value:.15f} ± {uncertainty:.15f}')

x = np.linspace(200,310)

plt.plot(x, params2[0]*np.exp(params2[1]*x),
        'r--',
        label="lineare Regression",
        linewidth=1)


plt.show()
plt.clf()

plt.plot(T1a, i1a-(params1[0]*np.exp(params1[1]*T1a)), "xb")
plt.plot(T1p, i1p-(params1[0]*np.exp(params1[1]*T1p)), "xr")
plt.xlim(200,310)
plt.ylim(-1,1.5)
plt.show()

plt.clf()

plt.plot(T2a, i2a-(params2[0]*np.exp(params2[1]*T2a)), "xb")
plt.plot(T2p, i2p-(params2[0]*np.exp(params2[1]*T2p)), "xr")
plt.xlim(200,320)
plt.ylim(-1,3)
plt.show()

plt.clf()

ab = T1an 
print(ab)

#bb = T1an
#print(ab)
#for i in range(len(ab)):
#    print(i)
#    print(ab[i])
#    bb[i] = ab[i] + 273.15
#print(ab)

reziprokT = 1/(ab)
MFL = np.log(i1an)

plt.plot(reziprokT, -MFL, "xk")

def sigmoid1(x, a, b):
    return a*(x)+b


params3, covariance_matrix = curve_fit(sigmoid1, reziprokT, MFL)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params:")
for name, value, uncertainty in zip('ab', params3, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

#x = np.linspace(-30,10)
x = np.linspace(0.0039,0.0043)

plt.plot(x, -(params3[0]*x + params3[1]) ,
        'k--',
        label="lineare Regression",
        linewidth=1)

#Params[0] -> a = -6716.6476 ± 400.5106 enstporicht W/(boltzmann) [also klevin]
#           ->    W = a * boltzmann 
k = const.Boltzmann
a = ufloat(params3[0], uncertainties[0])
W = a * -k
print("Aktivierungsenergie 1 celcisu pro minute, kleine Temperatur nährung, in eV: ")
print(W* 6.241509*10**18)
#
#

plt.show()

plt.clf()

ab = T2an 
print(ab)

#bb = T1an
#print(ab)
#for i in range(len(ab)):
#    print(i)
#    print(ab[i])
#    bb[i] = ab[i] + 273.15
#print(ab)

reziprokT = 1/(ab)
MFL = np.log(i2an)

plt.plot(reziprokT, -MFL, "xk")

def sigmoid1(x, a, b):
    return a*(x)+b


params3, covariance_matrix = curve_fit(sigmoid1, reziprokT, MFL)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params:")
for name, value, uncertainty in zip('ab', params3, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

#x = np.linspace(-30,10)
x = np.linspace(0.0039,0.0043)

plt.plot(x, -(params3[0]*x + params3[1]) ,
        'k--',
        label="lineare Regression",
        linewidth=1)


k = const.Boltzmann
a = ufloat(params3[0], uncertainties[0])
W = a * -k
print("Aktivierungsenergie 2 celcisu pro minute, kleine Temperatur nährung, in eV: ")
print(W* 6.241509*10**18)
#
#

plt.show()
plt.clf()


# Aktivierungsarbeit durch integration


