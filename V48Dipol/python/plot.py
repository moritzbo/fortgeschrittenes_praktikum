import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


# matplotlib.pyplot.rcdefaults()


t1a, T1ac, i1a = np.genfromtxt("data/A1TproM.dat", unpack=True)
t1p, T1pc, i1p = np.genfromtxt("data/Peaks1TproM.dat", unpack=True)
T1a= T1ac +273.15
T1p=T1pc + 273.15

t2a, T2ac, i2a = np.genfromtxt("data/A2TproM.dat", unpack=True)
t2p, T2pc, i2p = np.genfromtxt("data/Peaks2TproM.dat", unpack=True)
T2a= T2ac + 273.15
T2p= T2pc + 273.15

t1an, T1anc, i1an = np.genfromtxt("data/anlauf1TproM.dat", unpack=True)
t2an, T2anc, i2an = np.genfromtxt("data/anlauf2TproM.dat", unpack=True)
T1an= T1anc + 273.15
T2an= T2anc + 273.15

t1, Tneu1, dsfds = np.genfromtxt("data/1GRADproMIN.dat", unpack=True)
t2, Tneu2, fsddf = np.genfromtxt("data/2GRADproMIN.dat", unpack=True)
T1= Tneu1 + 273.15
T2= Tneu2 + 273.15




plt.plot(t1, T1, "og", markersize="1.5", label="Messdaten: 1° pro min")
plt.plot(t2, T2, "ob", markersize="1.5", label="Messdaten: 2° pro min")

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


plt.ylabel(r'T $[\si{\kelvin}]$')
plt.xlabel(r't $[\si{\minute}]$')

plt.grid()
plt.legend()
plt.savefig("build/Temperatur.pdf")
plt.clf()




plt.plot(T1a, i1a, "xb", label="Messdaten := Anstieg")
plt.plot(T1p, i1p, "xr", label="Messdaten := Entladungsstrom")  

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
        label="Ausgleichsfunktion",
        linewidth=1)

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

plt.grid()
plt.legend()
plt.savefig("build/Strom1.pdf")

plt.clf()
plt.plot(T2a, i2a, "xb", label="Messdaten := Anstieg")
plt.plot(T2p, i2p, "xr", label="Messdaten := Entladungsstrom") 

def sigmoid1(x, a, b):
    return a*np.exp(b*x)


params2, covariance_matrix = curve_fit(sigmoid1, T2a, i2a, p0=(1,0.01))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Untergrund2:")
for name, value, uncertainty in zip('ab', params2, uncertainties): 
    print(f'{name} = {value:.15f} ± {uncertainty:.15f}')

x = np.linspace(200,310)

plt.plot(x, params2[0]*np.exp(params2[1]*x),
        'k--',
        label="lineare Regression",
        linewidth=1)

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')
plt.grid()
plt.legend()
plt.savefig("build/Strom2.pdf")


plt.clf()

plt.plot(T1a, i1a-(params1[0]*np.exp(params1[1]*T1a)), "xb",label="Messdaten := Anstieg - Untergrund")
plt.plot(T1p, i1p-(params1[0]*np.exp(params1[1]*T1p)), "xr",label="Messdaten := Strom")  
plt.xlim(200,310)
plt.ylim(-1,1.5)

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

plt.grid()
plt.legend()
plt.savefig("build/untegrrund1.pdf")


plt.clf()

plt.plot(T2a, i2a-(params2[0]*np.exp(params2[1]*T2a)), "xb", label="Messdaten := Anstieg - Untergrund")
plt.plot(T2p, i2p-(params2[0]*np.exp(params2[1]*T2p)), "xr", label="Messdaten := Strom")  
plt.xlim(200,320)
plt.ylim(-1,3)



plt.ylabel(r'I $[\si{\pico\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

plt.grid()
plt.legend()
plt.savefig("build/untegrrund2.pdf")

plt.clf()

NFLStrom1 = dsfds - (params1[0]*np.exp(params1[1]*T1))
NFLStrom2 = fsddf - (params2[0]*np.exp(params2[1]*T2))


print(T1[27])
print(T1[54])
INT1 = 0

arrayIntvalues1 = []

array1 = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
for j in array1:
    for i in range(27 - array1.index(j)):
        INT1 = INT1 + (T1[i+j] - T1[i+j-1]) * (NFLStrom1[i+j] + NFLStrom1[i+j-1])/2
    arrayIntvalues1.append(INT1)
    INT1 = 0
print(f"HIER HIER HIER: {arrayIntvalues1}")

T1NEU = []
T2NEU = []
NFLStrom1NEU = []
NFLStrom2NEU = []

for m in range(27):
    T1NEU.append(T1[27+m])
    NFLStrom1NEU.append(NFLStrom1[27+m])

for n in range(18):
    T2NEU.append(T2[15+n])
    NFLStrom2NEU.append(NFLStrom2[27+m])

for i in range(27):
   INT1 = INT1 + (T1[i+28] - T1[i+27]) * (NFLStrom1[i+28] + NFLStrom1[i+27])/2
   print(INT1) 
print(f"HIER HIER HIER: {INT1:.2f}")

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

print(T2[15])
print(T2[33])
INT2 = 0
arrayIntvalues2 = []

array2 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
for j in array2:
    for i in range(18 - array2.index(j)):
        INT2 = INT2 + (T2[i+j] - T2[i+j-1]) * (NFLStrom2[i+j] + NFLStrom2[i+j-1])/2
    arrayIntvalues2.append(INT2)
    INT2 = 0
print(f"HIER HIER HIER: {arrayIntvalues2}")

INTWERT1 = np.log(np.array(arrayIntvalues1)/(np.array(NFLStrom1NEU) * 1))
INTWERT2 = np.log(np.array(arrayIntvalues2)/(np.array(NFLStrom2NEU) * 2))

def lol1(x, a, b):
    return a*x+b

params, covariance_matrix = curve_fit(lol1, 1/np.array(T1NEU), INTWERT1)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("YOYOOYOYOYOYOYOOYOYOYOYO:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0.00373,0.0042)

plt.plot(1/np.array(T1NEU), INTWERT1, "kx")

plt.plot(x, params[0]*x+params[1],
        'k--',
        
        linewidth=1)

#def lol2(x, a, b):
#    return a*x+b
#
#params, covariance_matrix = curve_fit(lol2, 1/np.array(T2NEU), INTWERT2)
#
#uncertainties = np.sqrt(np.diag(covariance_matrix))
#print("WASGEEEEEEEEEHT:")
#for name, value, uncertainty in zip('ab', params, uncertainties): 
#    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')
#
#x = np.linspace(240,280)
#
#plt.plot(x, params[0]*x+params[1],
#        'b--',
#        
#        linewidth=1)
#plt.plot(T2NEU, INTWERT2, "bx")

plt.savefig("build/lol1.pdf")
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
MFL = np.log(i1an-(params1[0]*np.exp(params1[1]*T1an)))
print("############################")
print(MFL)
plt.plot(reziprokT, -MFL, "xk", label="Messdaten")

def sigmoid1(x, a, b):
    return a*(x)+b


params3, covariance_matrix = curve_fit(sigmoid1, reziprokT, MFL)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params energie1:")
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

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

plt.grid()
plt.legend()
plt.savefig("build/benergie1.pdf")
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
MFL = np.log(i2an-(params2[0]*np.exp(params2[1]*T2an)))

plt.plot(reziprokT, -MFL, "xk", label="Messdaten")

def sigmoid1(x, a, b):
    return a*(x)+b


params3, covariance_matrix = curve_fit(sigmoid1, reziprokT, MFL)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params energie 2:")
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

plt.ylabel(r'I $10^{-11}$[$\si{\ampere}]$')
plt.xlabel(r'T $[\si{\kelvin}]$')

plt.grid()
plt.legend()
plt.savefig("build/benergie2.pdf")
plt.clf()


# Aktivierungsarbeit durch integration

