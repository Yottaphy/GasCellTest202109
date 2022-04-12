from math import erf, exp
from statistics import mode
import numpy as np
from numpy.lib.scimath import log, sqrt
import scipy as sc
from scipy.special import wofz
import sys
import csv
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit.models import SkewedVoigtModel

PI = 3.14159265359
c = 29.9792458 #cm/ns


#define some nicer format for the plots
plt.rcParams['font.size'] = 18

def z(x, mu, gamma, sigma):
    return (x-mu+ 1j*gamma)/(sigma*sqrt(2))

def voigt(x, a, mu, sigma, gamma):
    return a*np.real(wofz(z(x, mu, gamma, sigma)))/(sigma * sqrt(2*PI))

def jorge(x, a, mu, sigma, gamma, s):
    return (1+erf(s*(x-mu)/(sigma*sqrt(2))))*voigt(x, a, mu, sigma, gamma)
     
def sasha(x, A, B, nu0, gamma0, c, f):
    gamma = 2*gamma0/(1+exp(c*(x-nu0)))
    return f*(2*B)/(PI*gamma)/(1+4*((x-nu0)/gamma)**2) + (1-f)*A/gamma *sqrt(4*log(2)/PI)*exp(-4*log(2)*((x-nu0)/gamma)**2)

def FWHM(gamma, sigma):
    F1, F2, F3 = 1.0692, 0.8664, 5.545083
    return F1*gamma + sqrt(F2*gamma**2 + F3*sigma**2)

def sig(F, gamma):
    F1, F2, F3 = 1.0692, 0.8664, 5.545083
    return sqrt(((F-F1*gamma)**2 - F2*gamma**2)/F3)

def gam(F, l):
    F1, F2, F3 = 1.0692, 0.8664, 5.545083
    a = F2 + F3*l**2/(2*log(2)) -1 
    b = 2*F1
    c = -F**2
    return (-b+sqrt(b**2-4*a*c))/(2*a)

#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.
j = []
s = []
x = np.linspace(-40,40)

paramA  = -20679
paramB  = -33963
centre  = -1.98
width   = 79.6
skew    = 0.00189
lorentz = 19.2
gamma   = gam(width,lorentz)
sigma   = sig(width,gamma)
print(gamma, sigma, FWHM(gamma, sigma))

for n in x:
    #j.append(jorge(n, 5*paramA, centre, sigma, gamma, -2.5*skew))
    j.append(jorge(n, 20213.034610732186, 17.038149974440152-15, 7.006425008334865, 7.006425008334865, -0.20309462995313238))
    s.append(-sasha(n, paramA, paramB, centre, width, skew, lorentz)-400)

plt.plot(x, j, '-', color = '#ed5858',  label="Jorge")
plt.plot(x, s, '--', color = '#2c60a0',  label="Sasha*")

plt.grid(which='major', axis='both', linewidth=1)
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.legend(loc=1,fontsize= 'x-small')
# plt.yscale("log")
plt.savefig('difference.pdf', transparent=True, pad_inches='tight')


