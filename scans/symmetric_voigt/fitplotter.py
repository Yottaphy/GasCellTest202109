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
c  = 29.9792458 #cm/ns

def voigt(x, n, mu, sigma, gamma):
    return n*np.real(wofz((x-mu+1j*gamma)/sigma/sqrt(2)))/sigma/sqrt(2*PI)

def pol1(x, m, c):
    return m*x+c

def txtreader(filename):
    a,b,c,d,e,f,g,h,i,j,k = np.genfromtxt( filename, unpack= True, skip_header=1)
    return a,b,c,d,e,f,g,h,i,j,k

def main(stepno):
    print(stepno, "step:")
    anchor = (1,1)
    legendcols = 1

    pres, n, nerr, mu, muerr, fwhm, fwhmerr, sigma, sigmaerr, gamma, gammaerr = txtreader(str(stepno) + "step/stats"+str(stepno)+".txt")

    plt.rcParams['font.size'] = 18

    midpointwn = 10991.18507*2 if stepno == 2 else 13085.7509258*3
    muwn = []

    for i in range(len(mu)):
        muwn.append(mu[i] / c -midpointwn)

    if stepno == 2:
        xwn = np.linspace(-1,2,10000)
    else:
        xwn = np.linspace(-2,2,10000)
    

    for i in range(len(pres)):
        y = voigt(xwn, 1, muwn[i], sigma[i], gamma[i])
        plt.plot(xwn, y/max(y), '-', label = str(pres[i])+' mbar')

    plt.xlabel('Wavenumber - '+ str(round(midpointwn,1)) +' [cm$^{-1}$]')
    plt.ylabel('Intensity [arb. u.]')
    plt.title('Step '+str(stepno))
    plt.legend(fontsize= 'x-small', bbox_to_anchor = anchor, ncol = legendcols)
    plt.savefig("fits/allfits"+str(stepno)+"_waveno.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/allfits"+str(stepno)+"_waveno.png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

    if stepno == 2:
        midpointfreq = c*midpointwn
        x = np.linspace(6,25,10000)
    else:
        midpointfreq = c*midpointwn
        x = np.linspace(-12.5,-2.5,10000)

    for i in range(len(pres)):
        mu[i] -= midpointfreq
        
        y = voigt(x, 1, mu[i], sigma[i], gamma[i])
        plt.plot(x, y/max(y), '-', label = str(pres[i])+' mbar')

    plt.xlabel('Frequency - '+ str(round(midpointfreq,1)) +' [GHz]')
    plt.title('Step '+str(stepno))
    plt.ylabel('Intensity [arb. u.]')
    plt.legend(fontsize= 'x-small', bbox_to_anchor = anchor, ncol = legendcols)
    plt.savefig("fits/allfits"+str(stepno)+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/allfits"+str(stepno)+".png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()


    for i in range(len(mu)):
        mu[i]       *= 1000
        muerr[i]    *= 1000
        fwhm[i]     *= 1000
        fwhmerr[i]  *= 1000

    ext = np.arange(0,250,10)

    fit, err = curve_fit(pol1, pres, mu, absolute_sigma=False)
    if fit[1] < 0:
        equation1 = 'y = ' + str(round(fit[0],3))+'x '+str(round(fit[1],0))
    elif fit[1] == 0: 
        equation1 = 'y = ' + str(round(fit[0],3))+'x'
    else:
        equation1 = 'y = ' + str(round(fit[0],3))+'x +'+str(round(fit[1],0))
    plt.errorbar(pres, mu, yerr=muerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, pol1(ext, *fit), '-', color = 'C0')
    #plt.text(x= 0, y=min(mu), s= equation1)
    plt.grid(which='major', axis='both', linewidth=1)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Centroid Position \n- '+ str(round(midpointfreq*1000,0)) +'  [MHz]')
    plt.title('Pressure Shift - Step ' + str(stepno))
    plt.savefig("fits/shift"+str(stepno)+"_sigma.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/shift"+str(stepno)+"_sigma.png", bbox_inches = 'tight', pad_inches = 0.1)
    print("Shift = ", fit[0], sqrt(err[0][0]))
    plt.close()

    # for i in range(len(fwhm)):
    #     fwhm[i] = abs(fwhm[0] - fwhm[i])

    fit2, err2 = curve_fit(pol1, pres, fwhm, absolute_sigma=False)
    if fit2[1] < 0:
        equation2 = 'y = ' + str(round(fit2[0],2))+'x '+str(round(fit2[1],3))
    elif fit2[1] == 0: 
        equation2 = 'y = ' + str(round(fit2[0],2))+'x'
    else:
        equation2 = 'y = ' + str(round(fit2[0],2))+'x +'+str(round(fit2[1],3))
        
    for i in range(len(mu)):
        mu[i]       += midpointfreq
        mu[i]       *= 1000

    plt.errorbar(pres, fwhm, yerr=fwhmerr, marker='s', lw = 0, mfc = 'C0', ecolor = 'C0', elinewidth = 2)
    plt.plot(ext, pol1(ext, *fit2), '-', color = 'C0')
    #plt.text(x= 0, y=max(fwhm), s= equation2)
    plt.grid(which='major', axis='both', linewidth=1)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Voigt Fit FWHM [MHz]')
    plt.title('Pressure Broadening - Step ' + str(stepno))
    print("Broadening = ", fit2[0], sqrt(err2[0][0]))
    print("Width in Vacuum = ", fit2[1], sqrt(err2[1][1]))
    plt.savefig("fits/broadening"+str(stepno)+"_sigma.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("fits/broadening"+str(stepno)+"_sigma.png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

for stepno in [1,2]:
    main(stepno)