import numpy as np
from numpy.core.function_base import linspace
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

PI = 3.14159265359

def voigt(x, n, mu, sigma, gamma):
    return n*np.real(wofz((x-mu+1j*gamma)/sigma/sqrt(2)))/sigma/sqrt(2*PI)

def pol1(x, m, c):
    return m*x+c

def txtreader(filename):
    a,b,c,d,e,f,g,h,i,j,k = np.genfromtxt( filename, unpack= True, skip_header=1)
    return a,b,c,d,e,f,g,h,i,j,k

pres, n, nerr, mu, muerr, fwhm, fwhmerr, sigma, sigmaerr, gamma, gammaerr = txtreader("results/stats.txt")

x = np.linspace(-.5,.5,10000)
midpoint = 13085.683

plt.rcParams['font.size'] = 18

for i in range(len(pres)):
    mu[i] /= 29.9792458
    mu[i] -= midpoint
    muerr[i] /= 29.9792458
    plt.plot(x, voigt(x, 1, mu[i], sigma[i], gamma[i]), '-', label = str(pres[i])+' mbar')

plt.xlabel('Wavenumber from reference [cm$^{-1}$]')
plt.ylabel('Intensity [arb. u.]')
plt.title('Reference = 13085.683 cm$^{-1}$')
plt.legend(loc=2,fontsize= 'x-small')
plt.show()
plt.close()


for i in range(len(mu)):
    mu[i] += midpoint
    mu[i] *= 29.9792458
    muerr[i] *= 29.9792458
    mu[i] -= 39257.25

fit, err = curve_fit(pol1, pres, mu)
if fit[1] < 0:
    equation1 = 'y = ' + str(round(fit[0],5))+'x '+str(round(fit[1],3))
elif fit[1] == 0: 
    equation1 = 'y = ' + str(round(fit[0],5))+'x'
else:
    equation1 = 'y = ' + str(round(fit[0],5))+'x +'+str(round(fit[1],3))
plt.errorbar(pres, mu, yerr=muerr, marker='s', lw = 0, mfc = 'C0')
plt.plot(pres, pol1(pres, *fit), color = 'C0')
plt.text(x= 125, y=max(mu), s= equation1)
plt.grid(which='major', axis='both', linewidth=1)
plt.xlabel('Pressure [mbar]')
plt.ylabel('Centroid Position [GHz]')
plt.title('Pressure Shift')
plt.show()
plt.close()

# for i in range(len(fwhm)):
#     fwhm[i] = abs(fwhm[0] - fwhm[i])

fit2, err2 = curve_fit(pol1, pres, fwhm)
if fit2[1] < 0:
    equation2 = 'y = ' + str(round(fit2[0],5))+'x '+str(round(fit2[1],3))
elif fit2[1] == 0: 
    equation2 = 'y = ' + str(round(fit2[0],5))+'x'
else:
    equation2 = 'y = ' + str(round(fit2[0],5))+'x +'+str(round(fit2[1],3))

plt.errorbar(pres, fwhm, yerr=fwhmerr, marker='s', lw = 0, mfc = 'C0')
plt.plot(pres, pol1(pres, *fit2), color = 'C0')
plt.text(x= 50, y=max(fwhm), s= equation2)
plt.grid(which='major', axis='both', linewidth=1)
plt.xlabel('Pressure [mbar]')
plt.ylabel('Voigt Fit FWHM [GHz]')
plt.title('Pressure Broadening')
plt.show()