import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

start = time.clock()


def f2(y, t, extra):
    z, mp = init_common_params()

    m1 = np.array([y[0], y[1], y[2]])
    m2 = np.array([y[3], y[4], y[5]])

    a1, b1 = init_params1(t, extra)
    a2, b2 = init_params2(t, extra)

    h1, h2 = calc_h(m1[2], m2), calc_h(m2[2], m1)

    dm1_dt = np.cross(h1, m1) - a1 * np.cross(np.cross(h1, m1), m1) + b1 * np.cross(m1, np.cross(m1, mp))
    dm2_dt = np.cross(h2, m2) - a2 * np.cross(np.cross(h2, m2), m2) + b2 * np.cross(m2, np.cross(m2, mp))

    return np.concatenate((dm1_dt, dm2_dt), axis=None)


def init_y():
    e = 0.1
    yz = 1.-e
    y = np.array([0., np.sqrt(1.-yz*yz), yz])
    return np.concatenate((y, y), axis=None)


def init_common_params():
    z = np.array([0., 0., 1.])
    mp = np.array([0., 0., 1.])

    return z, mp


def calc_h(mz, m):
    h0 = 2.
    e = 0.001

    h = np.array([0., 0., h0 - mz]) + e*m
    return h


def init_params1(t, extra):
    a = 0.01
    b = 0.
    if t > N * S * 0.3:
        b = 0.019

    return a, b


def init_params2(t, extra):
    a = 0.01
    b = 0.
    if t > N * S * 0.3:
        b = 0.015

    return a, b


# debug boolean to activate prints
debug = True
# Number of points
N = 1000000
# Space between points
S = 0.01
# time vector
t = np.linspace(0., N * S, N)
# extra is not used here
extra = 0

# solved eq
Y = odeint(f2, init_y(), t, args=(extra,))

print(np.shape(Y))

if debug:
    print("last x1:", Y[(N - 10):, 0])
    print("last y1:", Y[(N - 10):, 1])
    print("last z1:", Y[(N - 10):, 2])
    print("last x2:", Y[(N - 10):, 3])
    print("last y2:", Y[(N - 10):, 4])
    print("last z2:", Y[(N - 10):, 5])
    plt.plot(t, Y)
    plt.legend(["x1", "y1", "z1", "x2", "y2", "z2"])
    plt.xlabel("Dimensionless time t")
    plt.ylabel("Normalized Magnetization m")
    plt.show()

# compute fourier transform of oscillation in x,y,z
# only take into account points after the cuton
cuton = int(N*0.9)
yfx1 = np.abs(np.fft.fft(Y[cuton:, 0]))
yfy1 = np.abs(np.fft.fft(Y[cuton:, 1]))
yfz1 = np.abs(np.fft.fft(Y[cuton:, 2]))
yfx2 = np.abs(np.fft.fft(Y[cuton:, 3]))
yfy2 = np.abs(np.fft.fft(Y[cuton:, 4]))
yfz2 = np.abs(np.fft.fft(Y[cuton:, 5]))

# compute frequency axis of fourier plot
freq = abs(np.fft.fftfreq(len(yfx1), S))

# find the freq of the maximum (without the first peak at 0)
# when amplitude is bigger than minAmp
minAmp = 1.

maxAmpX1 = np.amax(yfx1[1:])
oscAmpX1 = np.amax(Y[cuton:, 0])
fx1 = 0
if maxAmpX1 > minAmp:
    ix1 = np.where(yfx1 == maxAmpX1)
    fx1 = freq[ix1[0][0]]

maxAmpY1 = np.amax(yfy1[1:])
fy1 = 0
if maxAmpY1 > minAmp:
    iy1 = np.where(yfy1 == maxAmpY1)
    fy1 = freq[iy1[0][0]]

maxAmpZ1 = np.amax(yfz1[1:])
fz1 = 0
if maxAmpZ1 > minAmp:
    iz1 = np.where(yfz1 == maxAmpZ1)
    fz1 = freq[iz1[0][0]]

maxAmpX2 = np.amax(yfx2[1:])
oscAmpX2 = np.amax(Y[cuton:, 3])
fx2 = 0
if maxAmpX2 > minAmp:
    ix2 = np.where(yfx2 == maxAmpX2)
    fx2 = freq[ix2[0][0]]

maxAmpY2 = np.amax(yfy2[1:])
fy2 = 0
if maxAmpY2 > minAmp:
    iy2 = np.where(yfy2 == maxAmpY2)
    fy2 = freq[iy2[0][0]]

maxAmpZ2 = np.amax(yfz2[1:])
fz2 = 0
if maxAmpZ2 > minAmp:
    iz2 = np.where(yfz2 == maxAmpZ2)
    fz2 = freq[iz2[0][0]]

if debug:
    print("The frequency in x1 is:", fx1, "with amplitude:", maxAmpX1)
    print("The frequency in y1 is: ", fy1, "with amplitude:", maxAmpY1)
    print("The frequency in z1 is: ", fz1, "with amplitude:", maxAmpZ1)
    print("The frequency in x2 is:", fx2, "with amplitude:", maxAmpX2)
    print("The frequency in y2 is: ", fy2, "with amplitude:", maxAmpY2)
    print("The frequency in z2 is: ", fz2, "with amplitude:", maxAmpZ2)

    print("The oscillation amplitude in x1 is:", oscAmpX1)
    print("The oscillation amplitude in x2 is:", oscAmpX2)

    cutoff = N//1000
    plt.plot(freq[1:cutoff], yfx1[1:cutoff])
    plt.plot(freq[1:cutoff], yfy1[1:cutoff])
    plt.plot(freq[1:cutoff], yfz1[1:cutoff])
    plt.plot(freq[1:cutoff], yfx2[1:cutoff])
    plt.plot(freq[1:cutoff], yfy2[1:cutoff])
    plt.plot(freq[1:cutoff], yfz2[1:cutoff])

    plt.legend(["x1", "y1", "z1", "x2", "y2", "z2"])
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()