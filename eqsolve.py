import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

start = time.clock()


def f(y, t, extra):
    m = y
    mz = m[2]
    h, a, b, z, mp = init_params(t, extra, mz)
    return np.cross(h, m) - a * np.cross(np.cross(h, m), m) + b * np.cross(m, np.cross(m, mp))


def easy_f(y, t, extra):
    m = y
    h, a, b, z, mp = init_params(t, extra, m[2])
    return -h * np.cross(m, z) - a * np.cross(m, np.cross(m, z)) + b * np.cross(m, np.cross(m, mp))


def f2(y1, y2, t, extra):
    m1, m2 = y1, y2


def init_y():
    e = 0.1
    yz = 1.-e
    return np.array([0., np.sqrt(1.-yz*yz), yz])


def init_params(t, extra, mz):
    h0 = 2.
    e = 0.05
    w = 0.3 * 2 * np.pi

    h = np.array([0., 0., h0 - mz]) + e*np.array([np.sin(w*t), 0., 0.])

    a = 0.01
    b = 0.
    z = np.array([0., 0., 1.])
    mp = np.array([0., 0., 1.])

    if t > N*S*0.3:
        b = 0.0162

    return h, a, b, z, mp


# debug boolean to activate prints
debug = True
# Number of points
N = 100000
# Space between points
S = 0.1
# time vector
t = np.linspace(0., N * S, N)
# extra is not used here
extra = 0

# solved eq
Y = odeint(f, init_y(), t, args=(extra,))

end = time.clock()
print("time:", end - start)

if debug:
    print("last x:", Y[(N - 10):, 0])
    print("last y:", Y[(N - 10):, 1])
    print("last z:", Y[(N - 10):, 2])
    plt.plot(t, Y)
    plt.legend(["x", "y", "z"])
    plt.xlabel("Dimensionless time t")
    plt.ylabel("Normalized Magnetization m")
    plt.show()

# compute fourier transform of oscillation in x,y,z
# only take into account points after the cuton
cuton = int(N*0.9)
yfx = np.abs(np.fft.fft(Y[cuton:, 0]))
yfy = np.abs(np.fft.fft(Y[cuton:, 1]))
yfz = np.abs(np.fft.fft(Y[cuton:, 2]))

# compute frequency axis of fourier plot
freq = abs(np.fft.fftfreq(len(yfx), S))

# find the freq of the maximum (without the first peak at 0)
# when amplitude is bigger than minAmp
minAmp = 1.

maxAmpX = np.amax(yfx[1:])
fx = 0
if maxAmpX > minAmp:
    ix = np.where(yfx == maxAmpX)
    fx = freq[ix[0][0]]

maxAmpY = np.amax(yfy[1:])
fy = 0
if maxAmpY > minAmp:
    iy = np.where(yfy == maxAmpY)
    fy = freq[iy[0][0]]

maxAmpZ = np.amax(yfz[1:])
fz = 0
if maxAmpZ > minAmp:
    iz = np.where(yfz == maxAmpZ)
    fz = freq[iz[0][0]]

if debug:
    print("The frequency in x is:", fx, "with amplitude:", maxAmpX)
    print("The frequency in y is: ", fy, "with amplitude:", maxAmpY)
    print("The frequency in z is: ", fz, "with amplitude:", maxAmpZ)

    cutoff = N//1000
    plt.plot(freq[1:cutoff], yfx[1:cutoff])
    plt.plot(freq[1:cutoff], yfy[1:cutoff])
    plt.plot(freq[1:cutoff], yfz[1:cutoff])
    plt.legend(["x", "y", "z"])
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()