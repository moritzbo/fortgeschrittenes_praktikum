import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

W_1_1 = ufloat(0.64 , 0.05)* 1.602176634 * 10**(-19)
W_1_2 = ufloat(0.82 , 0.02)* 1.602176634 * 10**(-19)

W_2_1 = ufloat(0.79 , 0.02)* 1.602176634 * 10**(-19)
W_2_2 = ufloat(0.68 , 0.04)* 1.602176634 * 10**(-19)





Wmean1 = (W_1_1 + W_1_2) / 2
Wmean2 = (W_2_1 + W_2_2) / 2
theo = 0.66
abweichung1= (Wmean1/(1.602*10**(-19))-theo)/((Wmean1/(1.602*10**(-19))))*100
abweichung2= (Wmean2/(1.602*10**(-19))-theo)/((Wmean2/(1.602*10**(-19))))*100

print(abweichung1)
print(abweichung2)