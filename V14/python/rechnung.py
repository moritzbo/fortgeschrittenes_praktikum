import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

I, Ihülle, Iholz, Iallu, Ialles = np.genfromtxt("../data/alles.txt", unpack=True)

for i in range(len(I)):
    if Ihülle[i] == 0:
        print(I[i])