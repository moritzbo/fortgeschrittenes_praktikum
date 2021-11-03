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