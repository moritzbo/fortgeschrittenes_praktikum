import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp

c = const.physical_constants["speed of light in vacuum"]

c = c[0]

a = ufloat(22.860*10**(-3), 0.046*10**(-3))


lambda_g = 49*10**(-3)
print(c)

f = c * unp.sqrt((1/lambda_g)**2 + (1/(2*a))**2)


print(f)

vphase = f * lambda_g

print(vphase/c)


d1 = 63.5*10**(-3)
d2 = 61.8*10**(-3)

dist = d1 - d2

lambda_g_neu = 47.60*10**(-3)

SWR = np.sqrt(1+(1/(np.sin((np.pi*(dist))/(lambda_g_neu))**2)))

print(SWR)


SWR_neu = 10**((23)/20)

print(SWR_neu)