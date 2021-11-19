import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


plt.rcdefaults()

x = np.linspace(0,10)

plt.plot(x, x**(1/2),
            'k--',
            label="Messdaten",
            linewidth=1.5)

plt.savefig("build/plot1.pdf")