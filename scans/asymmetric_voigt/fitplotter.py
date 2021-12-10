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
from lmfit import Model
from lmfit.models import SkewedVoigtModel

PI = 3.14159265359
c  = 29.9792458 #cm/ns

def pol1(x, m, c):
    return m*x+c

def txtreader(filename):
    a,b,c,d,e = np.genfromtxt( filename, unpack= True, skip_header=1)
    return a,b,c,d,e

def main(stepno):
    print(stepno, "step:")
    anchor = (1,1)
    legendcols = 1

    pres, mu, muerr, fwhm, fwhmerr = txtreader(str(stepno) + "step/stats"+str(stepno)+"_asym.txt")

    plt.rcParams['font.size'] = 18
    
    midpoint = c*1E7/454.91 if stepno == 2 else c*1E7/254.73

    ext = np.arange(0,250,0.1)

    linfit = Model(pol1)
    params = linfit.make_params(m = -1, c = 0)
    
    plt.errorbar(pres, mu, yerr=muerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, pol1(ext, *fit), '-', color = 'red')
    #plt.text(x= 0, y=min(mu), s= equation1)
    # plt.grid(which='major', axis='both', linewidth=1)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    plt.xlim(0,250)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Centroid Position \n- '+ str(round(midpoint)) +'  [GHz]')
    # plt.title('Pressure Shift - Step ' + str(stepno))
    plt.savefig("fits/shift"+str(stepno)+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/shift"+str(stepno)+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
    print("Shift = ", fit[0], sqrt(err[0][0]))
    plt.close()

    fit2, err2 = curve_fit(pol1, pres, fwhm, sigma=fwhmerr, absolute_sigma=True)



    plt.errorbar(pres, fwhm, yerr=fwhmerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, pol1(ext, *fit2), '-', color = 'C0')
    #plt.text(x= 0, y=max(fwhm), s= equation2)
    #plt.grid(which='major', axis='both', linewidth=1)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    plt.xlim(0,250)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Voigt Fit FWHM [MHz]')
    plt.title('Pressure Broadening - Step ' + str(stepno))
    print("Broadening = ", fit2[0], sqrt(err2[0][0]))
    print("Width in Vacuum = ", fit2[1], sqrt(err2[1][1]))
    plt.savefig("fits/broadening"+str(stepno)+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/broadening"+str(stepno)+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

for stepno in [1,2]:
    main(stepno)