import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp
k = const.Boltzmann
Tmax1 = 255.45
Tmax2 = 260.35
b_1 = ufloat(1.009, 0.009)
b_2 = ufloat(2.050, 0.023)

W_1_1 = ufloat(0.64 , 0.05)* 1.602176634 * 10**(-19)
W_1_2 = ufloat(0.82 , 0.02)* 1.602176634 * 10**(-19)

W_2_1 = ufloat(0.79 , 0.02)* 1.602176634 * 10**(-19)
W_2_2 = ufloat(0.68 , 0.04)* 1.602176634 * 10**(-19)

tau_0_1_alpha = (k * (Tmax1)**2) / (W_1_1 * b_1) * unp.exp(- W_1_1/(k * Tmax1) )
tau_0_1_beta = (k * (Tmax1)**2) / (W_1_2 * b_1) * unp.exp(- W_1_2/(k * Tmax1) )

tau_0_2_alpha = (k * (Tmax2)**2) / (W_2_1 * b_2) * unp.exp(- W_2_1/(k * Tmax2) )
tau_0_2_beta =(k * (Tmax2)**2) / (W_2_2 * b_2) * unp.exp(- W_2_2/(k * Tmax2) )

print("01alpha")
print(f"{tau_0_1_alpha:.4g}")
print("01beta")
print(f'{tau_0_1_beta:.4g}')
print("02alpha")
print(f"{tau_0_2_alpha:.4g}")
print("02beta")
print(f"{tau_0_2_beta:.4g}")

tau1mean = (tau_0_1_alpha + tau_0_1_beta) / 2
tau2mean = (tau_0_2_alpha + tau_0_2_beta) /2


print(f"{tau1mean:.4g}")
print(f"{tau2mean:.4g}")

Wmean1 = (W_1_1 + W_1_2) / 2
Wmean2 = (W_2_1 + W_2_2) / 2

print(f"{Wmean1/(1.602*10**(-19)):.4g}")
print(f"{Wmean2/(1.602*10**(-19)):.4g}")



theo = 4*10**(-14)
abweichung1= (tau1mean-theo)/(tau1mean)*100
abweichung2= (tau2mean-theo)/(tau2mean)*100

print("########################")
print(abweichung1)
print(abweichung2)