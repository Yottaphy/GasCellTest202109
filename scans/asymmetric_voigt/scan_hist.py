from re import A
from statistics import mode
import numpy as np
from numpy.lib.scimath import log, sqrt
import scipy as sc
import scipy.optimize as opt
from scipy.special import wofz
import sys
import csv
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit.models import GaussianModel, SkewedVoigtModel, ConstantModel
import lmfit.lineshapes
from lmfit.lineshapes import voigt, erf
import scienceplots
from math import floor

plt.style.use("science")


def skewed_voigt2(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None, skew=0.0):
    """Return a Voigt lineshape, skewed with error function.
    Equal to: voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))
    where ``beta = skew/(sigma*sqrt(2))``
    with ``skew < 0``: tail to low value of centroid
         ``skew > 0``: tail to high value of centroid
    Useful, for example, for ad-hoc Compton scatter profile. For more
    information, see: https://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    beta = skew / max(1e-15, (2**0.5 * sigma))
    asym = 1 + erf(beta * (x - center))
    return (
        asym
        * amplitude
        * voigt(x, 1, center, sigma, gamma=gamma)
        / voigt(center, 1, center, sigma, gamma=gamma)
    )


PI = 3.14159265359
c = 29.9792458  # cm/ns


# read the file and output the correct count array and plot limits
def txtreader(filename):
    x, y, _ = np.genfromtxt(filename, unpack=True, delimiter=",", skip_header=1)
    return x, y


def txtreader2(filename):
    _, cts, _, _, wvno, _, _, _, _, _, _, _ = np.genfromtxt(
        filename, unpack=True, delimiter=",", skip_header=1
    )
    return wvno, cts


def rebin(x, y, newbin):
    x_step = (x.max() - x.min()) / (1 + newbin)
    newx = np.arange(0, newbin + 1) * x_step + x.min()
    newy = [0] * newbin

    j = 0
    for i in range(len(newx) - 1):
        if i == newbin:
            continue
        while x[j] <= newx[i + 1]:
            newy[i] += y[j]
            j += 1

    return newx, newy


# read 3 columns in order
in1 = sys.argv[1]
finalname = sys.argv[2]
stepno = sys.argv[3]
nbins = int(sys.argv[4])

# waveno, counts = txtreader(in1)
if stepno == "feb":
    waveno, counts = txtreader2(in1)
else:
    waveno, counts = txtreader(in1)

counterr = counts**0.5
# counterr = counts * 0.1
wght = (1 / counterr) ** 2
# counterr = [x*0.1 for x in counts]
# wght = [(1/x)**2 for x in counterr]

# define some nicer format for the plots


# FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.

# plot the experimental data (from the text file) as dots with error bars.
minimum = waveno.min()
maximum = waveno.max()
freq = []

if stepno == "2" or stepno == "feb":
    midpoint = c * 1e7 / 454.91
    for i in range(len(waveno)):
        freq.append(2 * c * waveno[i])
        freq[i] -= midpoint
elif stepno == "1":
    midpoint = c * 1e7 / 254.73
    for i in range(len(waveno)):
        freq.append(3 * c * waveno[i])
        freq[i] -= midpoint

freq = np.array(freq)
mask = np.bitwise_and.reduce([freq > -150, freq < 150])
freq = freq[mask]
counts = counts[mask]
counterr = counterr[mask]
wght = wght[mask]

print(freq)
freq.sort()

binned_freq, binned_counts = rebin(freq, counts, nbins)

bincentre = []
for i in range(1, len(binned_freq)):
    bincentre.append((binned_freq[i] + binned_freq[i - 1]) / 2)

print(bincentre)
print(binned_counts)

plt.stairs(
    binned_counts,
    binned_freq,
    color="grey",
    fill=True,
    label=finalname[:3].strip("m") + " mbar",
)

svm = SkewedVoigtModel()
svm.func = skewed_voigt2
# background = ConstantModel()
# svm = svm + background


params = svm.make_params(
    amplitude=200,
    center=-5,
    sigma=2,
    skew=-0.01,
    gamma=10,
    # c=-10,
)
params["gamma"].set(vary=True)
params["center"].set(vary=True, min=-100, max=100)
params["amplitude"].set(vary=True, min=0)
params["skew"].set(vary=True)
# params["c"].set(vary=True)
params["sigma"].set(vary=True)


result = svm.fit(binned_counts, params, x=bincentre, nan_policy="propagate")

print(result.fit_report())

step = 0.05
plot_x = np.arange(freq.min(), freq.max() + step, step)
plot_y = svm.eval(params=result.params, x=plot_x)
plt.plot(plot_x, plot_y, "-", color="orange", label="Asymmetric Voigt fit")
# plt.plot(freq, result.init_fit, '-', color = 'green',  label="Asymmetric Voigt fit")

plt.ylabel("Counts")
plt.xlabel("Frequency - " + str(round(midpoint)) + " [GHz]")
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.legend(loc=(0, 1.02), fontsize="small", ncol=2)
# plt.yscale("log")

# save plot as a pdf (vector images are superior, change my mind) with transparency
if stepno == "feb":
    plt.savefig(
        "histresults/feb_" + finalname + "_asym.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
if stepno == "2":
    plt.savefig(
        "histresults/2step_" + finalname + "_asym.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
elif stepno == "1":
    plt.savefig(
        "histresults/1step_" + finalname + "_asym.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
plt.close()

if stepno == "2":
    f = open("histresults/stats2_asym.txt", "a")
elif stepno == "1":
    f = open("histresults/stats1_asym.txt", "a")
elif stepno == "feb":
    f = open("histresults/statsfeb_asym.txt", "a")

a = result.params["amplitude"].value
aerr = result.params["amplitude"].stderr

g = result.params["gamma"].value
gerr = result.params["gamma"].stderr

s = result.params["sigma"].value
serr = result.params["sigma"].stderr

# F1   = 1.0692
# F2   = 0.8664
# F3   = 5.545083
# fwhm = F1*g + sqrt(F2*g**2 + F3*s**2)
# fwhmerr = sqrt((gerr*(F1+F2*g/(fwhm-F1*g)))**2 + (serr*s*F3/(fwhm-F1*g))**2)

mu = result.params["center"].value
muerr = result.params["center"].stderr
skew = result.params["skew"].value
skewerr = result.params["skew"].stderr

centre = plot_x[plot_y.argmax()]
centreerr = muerr

half_max = plot_y.max() / 2
ints = np.argwhere(np.diff(np.sign(plot_y - half_max)).flatten())
fwhm = float(plot_x[ints[1]] - plot_x[ints[0]])
fwhmerr = fwhm * sqrt((aerr / a) ** 2)

# plt.plot(plot_x[ints], plot_y[ints], 'go')

# f.write(finalname[0]+finalname[1]+finalname[2] + '\t\t\t\t\t\t' + str(result) + '\t' + str(sqrt(verr[0][0])) + '\t' + str(mean) + '\t' + str(meanerr) + '\t' + str(fwhm) + '\t' + str(fwhmerr) + '\t' + str(vfit[2])+ '\t' + str(verr[2][2]) + '\t' + str(vfit[3])+ '\t' + str(verr[3][3]) + '\n')

f.write(
    finalname[0]
    + finalname[1]
    + finalname[2]
    + "\t\t\t\t\t\t"
    + str(centre)
    + "\t"
    + str(centreerr)
    + "\t"
    + str(fwhm)
    + "\t"
    + str(fwhmerr)
    + "\n"
)

# plt.vlines(centre,0,500)
# plt.axvspan(freq[result.best_fit.argmax() -1], freq[result.best_fit.argmax()+2], color='red')


f.close()
