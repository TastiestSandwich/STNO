import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import random

start = time.clock()


def f2(y, t, extra1, extra2):
    # initiate stuff
    m = np.empty([n_stno, 3])
    b = np.empty([n_stno, 1])
    result = np.array([])

    # fill up m with 3d arrays for each stno
    for i, beta in enumerate(stno_b):
        m[i] = np.array([y[3*i], y[3*i+1], y[3*i+2]])
        b[i] = start_beta(t, beta)

    # fill up h with all m and result with dm/dt
    for i, mag in enumerate(m):
        h = calc_h(i, m, t, extra1, extra2)
        dmi_dt = np.cross(h, mag) - a * np.cross(np.cross(h, mag), mag) + b[i] * np.cross(mag, np.cross(mag, mp))
        result = np.concatenate((result, dmi_dt), axis=None)

    return result


def init_y():
    e = 0.1
    yz = 1.-e
    y = np.array([])
    # indexes 3i, 3i+1, 3i+2 are coordinates x, y & z from stno of index i
    for i in stno_b:
        y = np.concatenate((y, np.array([0., np.sqrt(1.-yz*yz), yz])), axis=None)
    return y


def calc_h(i, m, t, extra1, extra2):
    h0 = 2.
    ew1 = 0.01
    ew2 = 0.01
    w1 = extra1 * 2 * np.pi
    w2 = extra2 * 2 * np.pi
    mz = m[i, 2]

    # base external field h with external frequency w
    h = np.array([0., 0., h0 - mz]) + ew1*np.array([np.sin(w1*t), 0., 0.]) + ew2*np.array([np.sin(w2*t), 0., 0.])

    # add i to j stno interaction to h
    # for j, mag in enumerate(m):
    #     e = e_matrix[i][j]
    #     h = h + e*mag

    return h


def random_h():
    # generate random vector in unit cube
    v = np.random.uniform(-1., 1., 3)
    sq_dist = np.square(v[0]) + np.square(v[1]) + np.square(v[2])
    # discard vectors outside the unit sphere
    if sq_dist <= 1:
        return np.array(v)
    else:
        return random_h()


def init_common_params():
    z = np.array([0., 0., 1.])
    mp = np.array([0., 0., 1.])
    a = 0.01

    return z, mp, a


def calc_e_1d_closed(i, j):
    # 1d closed conditions
    # 1 - 2 - 3 - 4 - 1 - 2...
    e_first = 0.006
    e_second = 0.0015
    if i == j:
        return 0.
    elif (i-j) == 2 or (i-j) == -2:
        return e_second
    else:
        return e_first


def create_e_matrix_1d_open(n):
    # 1d open conditions
    # 1 - 2 - 3 - 4
    e_matrix = np.empty([n, n])
    e_first_neighbour = 0.006
    e_second_neighbour = 0.0015
    for i in range(n):
        for j in range(i+1):
            distance = i - j
            if distance == 0:
                e_matrix[i][j] = 0.
            elif distance == 1:
                e_matrix[i][j], e_matrix[j][i] = e_first_neighbour, e_first_neighbour
            elif distance == 2:
                e_matrix[i][j], e_matrix[j][i] = e_second_neighbour, e_second_neighbour
            else:
                e_matrix[i][j], e_matrix[j][i] = 0., 0.
    return e_matrix


def start_beta(t, beta):
    if t > N * S * 0.3:
        b = beta
    else:
        b = 0.

    return b


def create_dictionary():
    synch_dict = {}
    for i in range(n_stno):
        for j in freq_states:
            synch_dict[i, j] = []
    return synch_dict


def isclose(a, b):
    return np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)


# seed
np.random.seed(12345)
# Number of points
N = 1000000
# Space between points
S = 0.01
# time vector
t = np.linspace(0., N * S, N)

# n oscillators definition
stno_b = [0.015, 0.019, 0.023]
n_stno = len(stno_b)

# freq states definition
freq_states = ["A", "B", "AB"]

# initiate stuff
e_matrix = create_e_matrix_1d_open(n_stno)
print(e_matrix)
z, mp, a = init_common_params()

# values for frequency sweeping
extra_start = 0.2
extra_end = 0.4
extra_step = 0.01
extra_array = np.arange(extra_start, extra_end, extra_step)
extra_number = len(extra_array)

# create synch dictionary (freq vs stno)
synch_dict = create_dictionary()

i_array = range(0, extra_number)

for i1, extra1 in enumerate(extra_array):
    for i2, extra2 in enumerate(extra_array):
        print("freq1:", extra1)
        print("freq2:", extra2)
        # solved eq
        Y = odeint(f2, init_y(), t, args=(extra1, extra2))
        print("time needed: ", time.clock() - start)
        start = time.clock()

        # compute fourier transform of oscillation in x
        # only take into account points after the cuton
        cuton = int(N*0.9)
        yfx = np.empty([n_stno, N - cuton])
        for j in range(n_stno):
            # i*3 is x coordinate of stno index i (0, 3, 6...)
            yfx[j] = np.abs(np.fft.fft(Y[cuton:, j*3]))

        # compute frequency axis of fourier plot
        freq = abs(np.fft.fftfreq(N - cuton, S))

        # find the freq of the maximum (without the first peak at 0)
        # when amplitude is bigger than minAmp
        minAmp = 1.
        for j in range(n_stno):
            maxAmp = np.amax(yfx[j, 1:])
            stno_freq = 0.
            if maxAmp > minAmp:
                index = np.where(yfx[j] == maxAmp)
                stno_freq = freq[index[0][0]]

            # if stno is synched, store points in dictionary
            print(stno_freq)
            if isclose(stno_freq, extra1) and isclose(stno_freq, extra2):
                synch_dict[j, freq_states[2]].append((extra1, extra2))
                print("synch:", str(j + 1) + freq_states[2])
            elif isclose(stno_freq, extra1):
                synch_dict[j, freq_states[0]].append((extra1, extra2))
                print("synch:", str(j + 1) + freq_states[0])
            elif isclose(stno_freq, extra2):
                synch_dict[j, freq_states[1]].append((extra1, extra2))
                print("synch:", str(j + 1) + freq_states[1])
            else:
                print("synch:", "NONE")


end = time.clock()
print(end - start)

small_point_size = 2
big_point_size = 10

# plot fa vs fb with synched points
legend = []
for key, value in synch_dict.items():
    nk, fk = key
    legend.append(str(nk+1)+fk)
    if len(value) > 0:
        fa, fb = zip(*value)
        plt.scatter(fa, fb, big_point_size)

plt.legend(legend)
plt.xlabel("Frequency A")
plt.ylabel("Frequency B")
plt.show()