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


IallesFEHLER = np.array([])
IallesFEHLER = np.sqrt(Ialles)
# print(IallesFEHLER)
IallesFINAL = unp.uarray(Ialles,IallesFEHLER)

b = unp.log(IhülleRDY/IallesFINAL)
lul = b.reshape(12,1)
# print(b)
print(lul)

x = np.array([[14/41,-11/164,-13/82,(-2**0.5)/164,(132**0.5)/82,(-2**0.5)/164,-13/82,-11/164,14/41,(392**0.5)/328,(-152**0.5)/164,(-432**0.5)/328],[25/246,-16/123,25/246,(572**0.5)/328,(-32**0.5)/164,(-252**0.5)/328,-8/123,25/123,-8/123,(-252**0.5)/328,-32**0.5/164,(572**0.5)/328],[14/41,-11/164,-13/82,(-432**0.5)/328,(-152**0.5)/164,(392**0.5)/328,14/41,-11/164,-13/82,(-2**0.5)/164,(132**0.5)/82,(-2**0.5)/164],[-8/123,25/123,-8/123,(-252**0.5)/328,(-32**0.5)/164,(572**0.5)/328,25/246,-16/123,25/246,(-252**0.5)/328,(-32**0.5)/164,(572**0.5)/328],[-11/82,19/82,-11/82,(-2**0.5)/41,(112**0.5)/82,(-2**0.5)/41,-11/82,19/82,-11/82,-2**0.5/41,112**0.5/82,-2**0.5/41],[-8/123,25/123,-8/123,572**0.5/328,-32**0.5/164,-252**0.5/328,25/246,-16/123,25/246,572**0.5/328,-32**0.5/164,-252**0.5/328],[-13/82,-11/164,14/41,392**0.5/328,-152**0.5/164,-432**0.5/328,-13/82,-11/164,14/41,-2**0.5/164,132**0.5/82,-2**0.5/164],[25/246,-16/123,25/246,-252**0.5/328,-32**0.5/164,572**0.5/328,-8/123,25/123,-8/123,572**0.5/328,-32**0.5/164,-252**0.5/328],[-13/82,-11/164,14/41,-2**0.5/164,132**0.5/82,-2**0.5/164,14/41,-11/164,-13/82,-432**0.5/328,-152**0.5/164,392**0.5/328]])
print(x)
C = np.inner(b, x)

print(C)


ABallu = (müüü1-0.201)/0.201 *100
ABholz = (müüü2-0.052)/0.052 *100
ABblei = (müüü3-1.174)/1.174 *100
print(f"{ABallu:.4}")
print(f"{ABholz:.4}")
print(f"{ABblei:.4}")