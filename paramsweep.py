import numpy as np
import scipy as sp
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


def init_y():
    e = 0.1
    yz = 1. - e
    return np.array([0., np.sqrt(1. - yz * yz), yz])


def init_params(t, extra, mz):
    h0 = 2.
    e = 0.01
    w = 0.3 * 2 * np.pi

    h = np.array([0., 0., h0 - mz]) + e * np.array([np.sin(w * t), 0., 0.])

    a = 0.01
    b = 0.
    z = np.array([0., 0., 1.])
    mp = np.array([0., 0., 1.])

    if t > N * S * 0.3:
        b = extra

    return h, a, b, z, mp


# debug boolean to activate prints
debug = False
# Number of points
N = 1000000
# Space between points
S = 0.01
# time vector+
t = np.linspace(0., N * S, N)

# values for parameter sweeping
extra_start = 0.014
extra_end = 0.026
extra_step = 0.0002
extra_array = np.arange(extra_start, extra_end, extra_step)
extra_number = len(extra_array)

# frequency parameter sweep plot axes
x_freq_array = np.ones(extra_number)
y_freq_array = np.ones(extra_number)
z_freq_array = np.ones(extra_number)

# amplitude parameter sweep plot axes
x_amp_array = np.ones(extra_number)
y_amp_array = np.ones(extra_number)
z_amp_array = np.ones(extra_number)

i_array = range(0, extra_number)


for i in i_array:
    extra = extra_array[i]
    print(extra)
    # solved eq
    Y = odeint(f, init_y(), t, args=(extra,))

    if debug:
        plt.plot(t, Y)
        plt.legend(["x", "y", "z"])
        plt.show()

    # compute fourier transform of oscillation in x,y,z
    # only take into account points after the cuton
    cuton = int(N * 0.9)
    yfx = np.abs(np.fft.fft(Y[cuton:, 0]))
    yfy = np.abs(np.fft.fft(Y[cuton:, 1]))
    yfz = np.abs(np.fft.fft(Y[cuton:, 2]))

    # compute frequency axis of fourier plot
    freq = abs(np.fft.fftfreq(len(yfx), S))

    # find the freq of the maximum (without the first peak at 0)
    # when amplitude is bigger than minAmp
    minAmp = 1.

    maxAmpX = np.amax(yfx[1:])
    x_amp_array[i] = maxAmpX
    fx = 0.
    if maxAmpX > minAmp:
        ix = np.where(yfx == maxAmpX)
        fx = freq[ix[0][0]]
    x_freq_array[i] = fx

    maxAmpY = np.amax(yfy[1:])
    y_amp_array[i] = maxAmpY
    fy = 0.
    if maxAmpY > minAmp:
        iy = np.where(yfy == maxAmpY)
        fy = freq[iy[0][0]]
    y_freq_array[i] = fy

    maxAmpZ = np.amax(yfz[1:])
    z_amp_array[i] = maxAmpZ
    fz = 0.
    if maxAmpZ > minAmp:
        iz = np.where(yfz == maxAmpZ)
        fz = freq[iz[0][0]]
    z_freq_array[i] = fz

    if debug:
        print("The frequency in x is:", fx, "with amplitude:", maxAmpX)
        print("The frequency in y is: ", fy, "with amplitude:", maxAmpY)
        print("The frequency in z is: ", fz, "with amplitude:", maxAmpZ)

        plt.plot(freq[1:], yfx[1:])
        plt.plot(freq[1:], yfy[1:])
        plt.plot(freq[1:], yfz[1:])
        plt.show()


end = time.clock()
print(end - start)

point_size = 2
plt.scatter(extra_array, x_freq_array, point_size)
plt.scatter(extra_array, y_freq_array, point_size)
plt.scatter(extra_array, z_freq_array, point_size)
plt.legend(["x", "y", "z"])
plt.xlabel("Parameter β")
plt.ylabel("Frequency")
plt.show()

plt.plot(extra_array, x_amp_array)
plt.plot(extra_array, y_amp_array)
plt.plot(extra_array, z_amp_array)
plt.legend(["x", "y", "z"])
plt.xlabel("Parameter β")
plt.ylabel("Amplitude")
plt.show()
