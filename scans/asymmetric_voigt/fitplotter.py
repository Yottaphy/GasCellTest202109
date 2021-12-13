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
    shift = linfit.fit(mu, params, x= pres, weights= 1/muerr**2)
    shift_trend = shift.eval(shift.params, x=ext)

    equation_shift = '$\mu$ = ' + str(round(shift.params['m'].value, 4)) + 'P + ' + str(round(shift.params['c'].value,4))
    
    plt.errorbar(pres, mu, yerr=muerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, shift_trend, '-', color = 'red')
    plt.text(x= 10, y=min(mu)-1.5, s= equation_shift)
    # plt.grid(which='major', axis='both', linewidth=1)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    plt.xlim(0,250)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Centroid Position \n- '+ str(round(midpoint)) +'  [GHz]')
    plt.title('Pressure Shift - Step ' + str(stepno))
    plt.savefig("fits/shift"+str(stepno)+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/shift"+str(stepno)+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
    print("Shift = ", 1000*shift.params['m'].value, 1000*shift.params['m'].stderr)
    plt.close()

    broadening = linfit.fit(fwhm, params, x= pres, weights= 1/fwhmerr**2)
    broadening_trend = broadening.eval(broadening.params, x=ext)

    equation_broad = 'FWHM = ' + str(round(broadening.params['m'].value, 4)) + 'P + ' + str(round(broadening.params['c'].value,4))

    plt.errorbar(pres, fwhm, yerr=fwhmerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, broadening_trend, '-', color = 'red')
    plt.text(x= 10, y=max(fwhm)+5, s= equation_broad)
    #plt.grid(which='major', axis='both', linewidth=1)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    plt.xlim(0,250)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Voigt Fit FWHM [MHz]')
    plt.title('Pressure Broadening - Step ' + str(stepno))
    print("Broadening = ", 1000*broadening.params['m'].value, 1000*broadening.params['m'].stderr)
    print("Width in Vacuum = ", broadening.params['c'].value, broadening.params['c'].stderr)
    plt.savefig("fits/broadening"+str(stepno)+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/broadening"+str(stepno)+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

for stepno in [1,2]:
    main(stepno)