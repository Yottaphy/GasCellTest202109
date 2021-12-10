import satlas2 as sat
import numpy as np
import matplotlib.pyplot as plt
import time

def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input<0] = 1e-12
    return output

def beta(mass, V):
    r"""Calculates the beta-factor for a mass in amu
    and applied voltage in Volt. The formula used is

    .. math::

        \beta = \sqrt{1-\frac{m^2c^4}{\left(mc^2+eV\right)^2}}

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        voltage in volt.

    Returns
    -------
    float
        Relativistic beta-factor.
    """
    c = 299792458.0
    q = 1.60217657 * (10 ** (-19))
    AMU2KG = 1.66053892 * 10 ** (-27)
    mass = mass * AMU2KG
    top = mass ** 2 * c ** 4
    bottom = (mass * c ** 2 + q * V) ** 2
    beta = np.sqrt(1 - top / bottom)
    return beta

def dopplerfactor(mass, V):
    r"""Calculates the Doppler shift of the laser frequency for a
    given mass in amu and voltage in V. Transforms from the lab frame
    to the particle frame. The formula used is

    .. math::

        doppler = \sqrt{\frac{1-\beta}{1+\beta}}

    To invert, divide instead of multiply with
    this factor.

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        Voltage in volt.

    Returns
    -------
    float
        Doppler factor.
    """
    betaFactor = beta(mass, V)
    dopplerFactor = np.sqrt((1.0 - betaFactor) / (1.0 + betaFactor))
    return dopplerFactor

def loadData(run=1):
    calib_1 = np.loadtxt('Satlas test data/calib_{}.csv'.format(run), delimiter=',', skiprows=2)
    setpoints = calib_1[:, 0]
    voltage = calib_1[:, 1] * 1000

    A = np.vstack([setpoints, np.ones(setpoints.shape[0])]).T
    m, c = np.linalg.lstsq(A, voltage, rcond=None)[0]
    data = np.loadtxt('Satlas test data/Run_{}.csv'.format(run), delimiter=',')
    voltage = data[:, 1]
    dt = data[:, 4]
    if run >  1:
        mask = np.logical_and.reduce([dt > 100, dt < 120])
        voltage = voltage[mask]
    voltage = sorted(voltage)
    voltage, counts = np.unique(voltage, return_counts=True)
    err = np.sqrt(counts)
    err[np.isclose(err, 0)] = 1
    with open('Satlas test data/calib_{}.csv'.format(run)) as f:
        for i, line in enumerate(f):
            if i == 0:
                coolerVoltage = float(line.split(':')[1]) * 10000
            elif i == 1:
                wavenumber = float(line.split(':')[1])
                break
    return (wavenumber, coolerVoltage, m, c), (voltage, counts, err)

def make_transform(wavenumber, coolerVoltage, m, c, mass, center=None):
    freq = wavenumber * 29979.2458
    def transform(voltage):
        voltage = m * voltage + c
        V = coolerVoltage - voltage
        if center is None:
            return 2*freq / dopplerfactor(mass, V) - freq
        else:
            return 2*freq / dopplerfactor(mass, V) - center * 29979.2458
    return transform

f = sat.Fitter()
data = np.loadtxt('50mbar_1step_gascell.csv', delimiter=',', skiprows=1)
data[:, 0] = data[:, 0] * 29979.2458
data[:, 0] -= data[:, 0].mean()
data = sat.Source(data[:, 0], data[:, 1], yerr=modifiedSqrt, name='mbar50')
f.addSource(data)

# voigt = sat.Voigt(600, 1000, 500, 500, name='Symmetric_Peak')
voigt = sat.SkewedVoigt(600, 1000, 500, 500, 0, name='Asymmetric_Peak')
voigt.setVary('Skew', True)
data.addModel(voigt)
background = sat.Polynomial([30], name='Background')
data.addModel(background)
# f.fit()
# f.fit(method='emcee', llh_method='poisson', steps=5000, filename='50mbar.h5')
f.readWalk("50mbar.h5")
fwhm = voigt.calculateFWHM()

sat.generateCorrelationPlot("50mbar.h5", selection=(10, 100))
sat.generateWalkPlot("50mbar.h5", selection=(10, 100))
print(f.reportFit())
print('FWHM: {:.3f}+/-{:.3f}'.format(*fwhm))

plot_x = np.linspace(data.x.min(), data.x.max(), 4000)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.errorbar(data.x, data.y, modifiedSqrt(data.y), fmt='o', zorder=0)
ax.plot(plot_x, data.evaluate(plot_x), zorder=1)
plt.show()
