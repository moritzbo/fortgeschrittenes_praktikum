import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

I, Ihülle, Iholz, IIDK, Ialles = np.genfromtxt("../data/alles.txt", unpack=True)

IhülleNEU = np.array([])
IholzNEU = np.array([])
IIDKNEU = np.array([])

def test1(x):
    if Ihülle[x] > 0:
        global IhülleNEU 
        IhülleNEU = np.append(IhülleNEU, Ihülle[x])
        x=x+1
        test1(x)

test1(0)

def test2(x):
    if Iholz[x] > 0:
        global IholzNEU 
        IholzNEU = np.append(IholzNEU, Iholz[x])
        x=x+1
        test2(x)

test2(0)

def test3(x):
    if IIDK[x] > 0:
        global IIDKNEU 
        IIDKNEU = np.append(IIDKNEU, IIDK[x])
        x=x+1
        test3(x)

test3(0)

IhülleRDY = ufloat(np.mean(IhülleNEU),np.std(IhülleNEU))
IholzRDY = ufloat(np.mean(IholzNEU),np.std(IholzNEU))
IIDKRDY = ufloat(np.mean(IIDKNEU),np.std(IIDKNEU))

# print(IhülleRDY)
# print(IholzRDY)
# print(IIDKRDY)

müüü1 = -1 *unp.log(IhülleRDY/4765)*1/(0.2)
print(müüü1)
müüü2 = -1 *unp.log(IholzRDY/IhülleRDY)*1/(3)
print(müüü2)
müüü3 = -1 *unp.log(IIDKRDY/IhülleRDY)*1/(3)
print(müüü3)

