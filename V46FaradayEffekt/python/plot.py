import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp



Index16, Minimum16, WinkelGrad16 , WinkelMinute16, Filter = np.genfromtxt("data/GaAsN16.dat", unpack=True)
Index28, Minimum28, WinkelGrad28 , WinkelMinute28, Filter = np.genfromtxt("data/GaAsN28.dat", unpack=True)
IndexR,  MinimumR,  WinkelGradR,   WinkelMinuteR,  Filter = np.genfromtxt("data/GaAsREIN.dat", unpack=True)
tesla, Abstand = np.genfromtxt("data/bfeld.dat", unpack=True)

LR  = 5.11e-3
L16 = 1.36e-3
L28 = 1.296e-3


# Minuten in Grad
# 60min = 1grad
# 1 min = 1/60 grad 
WinkelMinuteInGrad16 = WinkelMinute16 * 1/60
WinkelMinuteInGrad28 = WinkelMinute28 * 1/60
WinkelMinuteInGradR = WinkelMinuteR* 1/60

Winkel16 = WinkelGrad16 + WinkelMinuteInGrad16
Winkel28 = WinkelGrad28 + WinkelMinuteInGrad28
WinkelR = WinkelGradR + WinkelMinuteInGradR
 

plt.plot(Abstand, tesla, "kx", label="Messwerte")

plt.ylabel(r'$B$ [$\si{\milli\tesla}]$')
plt.xlabel(r'$x$ [$\si{\milli\meter}]$')

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("build/plot1.pdf")

plt.clf()


#Drehwinkel bestimmen
a = len(WinkelR)/2
WinkelRSumme = []
FilterI = []

for i in range(0, len(WinkelR), 2):
    FilterI.append(Filter[i])
    
#print(FilterI)

def loopit(i,j):
    if i < len(WinkelR):
        WinkelRSumme.append(((WinkelR[i]-WinkelR[i+1])**2)**(1/2)/2)
        #print(WinkelRSumme[j])
        j = j+1
        i = i+2
        loopit(i,j)

print(loopit(0,0))

print(WinkelRSumme)

WinkelRarray = np.array(WinkelRSumme)
WinkelRnor = WinkelRarray/(LR)

BS = np.array(FilterI)
print(BS)

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1,BS*BS, WinkelRnor)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg für REINE PROBE:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,7.3)

#plt.plot(x, params[0]*x+params[1], 
#        'r--',
#        label="Messwerte, rein",
#        linewidth=1)



plt.plot(BS*BS, np.deg2rad(WinkelRnor), "rx", label="Messwerte, rein")
#plt.show()
#plt.clf()

a = len(Winkel16)/2
Winkel16Summe = []
FilterI = [] 



for i in range(0, len(Winkel16), 2):
    FilterI.append(Filter[i])
    
#print(FilterI)

def loopit(i,j):
    if i < len(Winkel16):
        Winkel16Summe.append(((Winkel16[i]-Winkel16[i+1])**2)**(1/2)/2)
        #print(Winkel16Summe[j])
        j = j+1
        i = i+2 
        loopit(i,j)

print(loopit(0,0))

print(FilterI)
print(Winkel16Summe)

Winkel16array = np.array(Winkel16Summe)
Winkel16nor = Winkel16array/(L16)

BS = np.array(FilterI)
print(BS)

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1,BS*BS, Winkel16nor)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg für UNREIN 1,2:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,7.3)

#plt.plot(x, params[0]*x+params[1], 
#        'k--',
#        label="Messwerte, dotiert 1.2",
#        linewidth=1)


plt.plot(BS*BS, np.deg2rad(Winkel16nor), "kx",  label="Messwerte, dotiert 1.2")
#plt.show()
#plt.clf()

a = len(Winkel28)/2
Winkel28Summe = []
FilterI = []

for i in range(0, len(Winkel28), 2):
    FilterI.append(Filter[i])
    
#print(FilterI)

def loopit(i,j):
    if i < len(Winkel28):
        Winkel28Summe.append(((Winkel28[i]-Winkel28[i+1])**2)**(1/2)/2)
        #print(Winkel28Summe[j])
        j = j+1
        i = i+2
        loopit(i,j)

print(loopit(0,0))

print(FilterI)
print(Winkel28Summe)

Winkel28array = np.array(Winkel28Summe)
Winkel28nor = Winkel28array/(L28)


BS = np.array(FilterI)
print(BS)

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1,BS*BS, Winkel28nor)

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg für UNREINE 2.8:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,7.3)

# plt.plot(x, params[0]*x+params[1], 
        # 'b--',
        # label="Messwerte, dotiert 2.8",
        # linewidth=1)

BS = np.array(FilterI)
print(BS)
plt.plot(BS*BS, np.deg2rad(Winkel28nor), "bx",
label="Messwerte, dotiert 2.8",)

plt.ylabel(r'$\theta_{\text{norm}}$ [$\si{\radian\per\meter}]$')
plt.xlabel(r'$\lambda^2$ [$\si{\micro\meter^2}]$')

plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig("build/plot2.pdf")
plt.clf()


#### NORMIERUNG DER WINKEL --> /L

LR  = 5.11e-3
L16 = 1.36e-3
L28 = 1.296e-3


#### bestimmung effektiver Masse 

WinkelDifferenzR16 = np.abs(WinkelRnor - Winkel16nor)
WinkelDifferenzR28 = np.abs(WinkelRnor - Winkel28nor)



BS = np.array(FilterI)
#BS = BS1*10**(-6)

plt.plot(BS*BS, np.deg2rad(WinkelDifferenzR16), "ko",label="Winkeldifferenz")

def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1,BS*BS, np.deg2rad(WinkelDifferenzR16))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg für Differenz REIN und 1.2:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,7.3)

plt.plot(x, params[0]*x+params[1], 
        'k--',
        label="lineare Regression",
        linewidth=1)

plt.grid()
plt.legend()

plt.ylabel(r'$\theta_{\text{frei}}$ [$\si{\radian\per\meter}]$')
plt.xlabel(r'$\lambda^2$ [$\si{\micro\meter^2}]$')
plt.tight_layout()
plt.savefig("build/plot3.pdf")
plt.clf()

#params[0] = ((const.elemenerycharge**3*N*B)/(n*8*const.pi**2*const.epsilonnull*const.speedoflight**3*m**2))

N = 1.2e24
B = 408e-3
n = 3.87 #!!!!!!!!!!

params12ERR0 = ufloat(params[0], uncertainties[0])
params12ERR1 = ufloat(params[1], uncertainties[1])
print(params12ERR0)

m1 = ((const.e**3)*N*B)
m2 = (n*8*(const.pi**2)*const.epsilon_0*(const.c**3)*(params12ERR0*10**(12)))
print("hier soll die masse stehen 1.2:")
print(f'{(m1/m2)**(1/2):.4}')
verhältnis1 = ((m1/m2)**(1/2))/const.m_e
print(verhältnis1)




plt.plot(BS*BS, np.deg2rad(WinkelDifferenzR28), "bo", label="Winkeldifferenz")



def sigmoid1(x, a, b):
    return a*x+b


params, covariance_matrix = curve_fit(sigmoid1,BS*BS, np.deg2rad(WinkelDifferenzR28))

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("Params Anstieg für Differenz REIN und 2.8:")
for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:.4f} ± {uncertainty:.4f}')

x = np.linspace(0,7.3)

plt.plot(x, params[0]*x+params[1], 
        'b--',
        label="lineare Regression",
        linewidth=1)

plt.grid()
plt.legend()

plt.ylabel(r'$\theta_{\text{frei}}$ [$\si{\radian\per\meter}]$')
plt.xlabel(r'$\lambda^2$ [$\si{\micro\meter^2}]$')
plt.tight_layout()

plt.savefig("build/plot4.pdf")

N = 2.8e24
B = 408e-3
n = 3.56 #!!!!!!!!!!

params28ERR0 = ufloat(params[0], uncertainties[0])
params28ERR1 = ufloat(params[1], uncertainties[1])

m1 = ((const.e**3)*N*B)
m2 = (n*8*(const.pi**2)*const.epsilon_0*(const.c**3)*(params28ERR0*10**(12)))
print("hier soll die masse stehen 2.8:")
print(f'{(m1/m2)**(1/2):.4}')
verhältnis2 = ((m1/m2)**(1/2))/const.m_e
print(verhältnis2)

print("########### BITTE HIER SCHAUEN")
print(WinkelDifferenzR16)
print(WinkelDifferenzR28)

print("########### HIER IN RADIANT")
print(np.deg2rad(WinkelDifferenzR16))
print(np.deg2rad(WinkelDifferenzR28)) 

print("############ ABWEICHUNGEN:")
theorie  = 0.06
abweichung1 = (verhältnis1 - theorie)/(theorie)*100
abweichung2 = (verhältnis2 - theorie)/(theorie)*100
print(f'{abweichung1:.4}')
print(f'{abweichung2:.4}')

print("NUR SPAßlol")