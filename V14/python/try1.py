import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
import uncertainties.unumpy as unp


x1= ufloat(0.075, 0.030)      
x2= ufloat(1.110, 0.029)      
x3= ufloat(0.064, 0.035)      
x4= ufloat(0.024, 0.039)      
x5= ufloat(1.095, 0.028)      
x6= ufloat(0.032, 0.041)      
x7= ufloat(0.116, 0.042)      
x8= ufloat(1.046, 0.028)      
x9= ufloat(0.118, 0.042) 


holz = 0.052
blei = 1.174


x1proz = unp.sqrt((x1 - holz)**2)/(holz)
x2proz = unp.sqrt((x2 - blei)**2)/(blei)
x3proz = unp.sqrt((x3 - holz)**2)/(holz)
x4proz = unp.sqrt((x4 - holz)**2)/(holz)
x5proz = unp.sqrt((x5 - blei)**2)/(blei)
x6proz = unp.sqrt((x6 - holz)**2)/(holz)
x7proz = unp.sqrt((x7 - holz)**2)/(holz)
x8proz = unp.sqrt((x8 - blei)**2)/(blei)
x9proz = unp.sqrt((x9 - holz)**2)/(holz)
print(x1proz)
print(x2proz)
print(x3proz)
print(x4proz)
print(x5proz)
print(x6proz)
print(x7proz)
print(x8proz)
print(x9proz)
