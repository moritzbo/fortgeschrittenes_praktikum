import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp



Index16, Minimum16, WinkelGrad16 , WinkelMinute16, Filter = np.genfromtxt("../data/GaAsN16.dat", unpack=True)
Index28, Minimum28, WinkelGrad28 , WinkelMinute28, Filter = np.genfromtxt("../data/GaAsN28.dat", unpack=True)
IndexR,  MinimumR,  WinkelGradR,   WinkelMinuteR,  Filter = np.genfromtxt("../data/GaAsREIN.dat", unpack=True)
tesla, Abstand = np.genfromtxt("../data/bfeld.dat", unpack=True)

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
plt.legend()
plt.grid()
plt.show()

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

print(FilterI)
print(WinkelRSumme)

BS = np.array(FilterI)
print(BS)
plt.plot(BS*BS, WinkelRSumme, "kx")
plt.show()
plt.clf()

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

BS = np.array(FilterI)
print(BS)
plt.plot(BS*BS, Winkel16Summe, "kx")
plt.show()
plt.clf()

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

BS = np.array(FilterI)
print(BS)
plt.plot(BS*BS, Winkel28Summe, "kx")
plt.show()
plt.clf()