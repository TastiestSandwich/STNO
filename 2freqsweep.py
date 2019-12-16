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

    h1, h2 = calc_h(m1[2], m2, t, extra), calc_h(m2[2], m1, t, extra)

    dm1_dt = np.cross(h1, m1) - a1 * np.cross(np.cross(h1, m1), m1) + b1 * np.cross(m1, np.cross(m1, mp))
    dm2_dt = np.cross(h2, m2) - a2 * np.cross(np.cross(h2, m2), m2) + b2 * np.cross(m2, np.cross(m2, mp))

    result = np.concatenate((dm1_dt, dm2_dt), axis=None)
    return result


def init_y():
    e = 0.1
    yz = 1.-e
    y = np.array([0., np.sqrt(1.-yz*yz), yz])
    return np.concatenate((y, y), axis=None)


def calc_h(mz, m, t, extra):
    h0 = 2.
    e = 0.003
    ew = 0.03
    w = extra * 2 * np.pi

    h = np.array([0., 0., h0 - mz]) + e*m + ew*np.array([np.sin(w*t), 0., 0.])
    return h


def init_common_params():
    z = np.array([0., 0., 1.])
    mp = np.array([0., 0., 1.])

    return z, mp


def init_params1(t, extra):
    a = 0.01
    b = 0.
    if t > N * S * 0.3:
        b = 0.015

    return a, b


def init_params2(t, extra):
    a = 0.01
    b = 0.
    if t > N * S * 0.3:
        b = 0.025

    return a, b


# debug boolean to activate prints
debug = False
# Number of points
N = 1000000
# Space between points
S = 0.01
# time vector
t = np.linspace(0., N * S, N)

# values for parameter sweeping
extra_start = 0.
extra_end = 0.5
extra_step = 0.01
extra_array = np.arange(extra_start, extra_end, extra_step)
extra_number = len(extra_array)

# frequency parameter sweep plot axes
x1_freq_array = np.ones(extra_number)
x2_freq_array = np.ones(extra_number)
x1_amp_array = np.ones(extra_number)
x2_amp_array = np.ones(extra_number)

i_array = range(0, extra_number)

for i in i_array:
    extra = extra_array[i]
    print(extra)
    # solved eq
    Y = odeint(f2, init_y(), t, args=(extra,))
    print(Y.shape)

    if debug:
        plt.plot(t, Y)
        plt.legend(["x1", "y1", "z1", "x2", "y2", "z2"])
        plt.xlabel("Dimensionless time t")
        plt.ylabel("Normalized Magnetization m")
        plt.show()

    # compute fourier transform of oscillation in x,y,z
    # only take into account points after the cuton
    cuton = int(N*0.9)
    yfx1 = np.abs(np.fft.fft(Y[cuton:, 0]))
    yfx2 = np.abs(np.fft.fft(Y[cuton:, 3]))

    # compute frequency axis of fourier plot
    freq = abs(np.fft.fftfreq(len(yfx1), S))

    # find the freq of the maximum (without the first peak at 0)
    # when amplitude is bigger than minAmp
    minAmp = 1.

    # amplitude and frequency for x1
    maxAmpX1 = np.amax(yfx1[1:])
    oscAmpX1 = np.amax(Y[cuton:, 0])
    fx1 = 0
    if maxAmpX1 > minAmp:
        ix1 = np.where(yfx1 == maxAmpX1)
        fx1 = freq[ix1[0][0]]
    x1_freq_array[i] = fx1
    x1_amp_array[i] = oscAmpX1

    # amplitude and frequency for x2
    maxAmpX2 = np.amax(yfx2[1:])
    oscAmpX2 = np.amax(Y[cuton:, 3])
    fx2 = 0
    if maxAmpX2 > minAmp:
        ix2 = np.where(yfx2 == maxAmpX2)
        fx2 = freq[ix2[0][0]]
    x2_freq_array[i] = fx2
    x2_amp_array[i] = oscAmpX2

    if debug:
        print("The frequency in x1 is:", fx1, "with amplitude:", maxAmpX1)
        print("The frequency in x2 is:", fx2, "with amplitude:", maxAmpX2)

        cutoff = N//1000
        plt.plot(freq[1:cutoff], yfx1[1:cutoff])
        plt.plot(freq[1:cutoff], yfx2[1:cutoff])

        plt.legend(["STNO 1", "STNO 2"])
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.show()

end = time.clock()
print(end - start)

point_size = 4
plt.scatter(extra_array, x1_freq_array, point_size)
plt.scatter(extra_array, x2_freq_array, point_size)
plt.scatter(extra_array, extra_array, 2)
plt.legend(["STNO 1", "STNO 2", "External Frequency"])
plt.xlabel("External Frequency")
plt.ylabel("Frequency")
plt.show()

plt.scatter(extra_array, x1_amp_array, point_size)
plt.scatter(extra_array, x2_amp_array, point_size)
plt.legend(["STNO 1", "STNO 2"])
plt.xlabel("External Frequency")
plt.ylabel("Amplitude")
plt.show()