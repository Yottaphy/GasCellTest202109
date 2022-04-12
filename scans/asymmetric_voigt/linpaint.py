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

def linfit(x, m, c):
    return m*x+c

#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.
xfeb = [97, 133, 170, 203]

jfeb = [12.082906763534993, -4.334352628327906, -6.480284514720552, -9.0819194924552]
jfeb = [x-15 for x in jfeb]
jfeberr = [0.9014539474834994, 1.008839022512276, 0.7185942160013863, 1.9712882200656767]

jfit = []
for x in xfeb:
    jfit.append(linfit(x, -0.1859, 26.0649-15))

sfeb = [-11.66662,-16.098,-20.272,-28.80]
sfeberr = [0.1982,0.2229,0.2741,0.51082]
sfit = []
for x in xfeb:
    sfit.append(linfit(x, -0.13654, 1.8602))
sfit2 = []
for x in xfeb:
    sfit2.append(linfit(x, -0.11845, -0.2247))

plt.errorbar(xfeb, jfeb, yerr=jfeberr, fmt='o', color = '#ed5858',  label="Jorge points")
plt.plot(xfeb, jfit, '-', color = '#ed5858',  label="Jorge fit")
plt.plot(xfeb, sfit, '--', color = '#2c60a0',  label="Sasha fit all")
plt.plot(xfeb, sfit2, '-', color = '#2c60a0',  label="Sasha fit excluding last")
plt.errorbar(xfeb, sfeb, yerr=sfeberr, fmt='^', color = '#2c60a0',  label="Sasha points")


plt.grid(which='major', axis='both', linewidth=1)
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.xlabel("Frequency from reference (GHz)")
plt.ylabel("Counts")
plt.legend(loc=1,fontsize= 'x-small')
# plt.yscale("log")
plt.savefig('difference.pdf', transparent=True, bbox_inches='tight')


