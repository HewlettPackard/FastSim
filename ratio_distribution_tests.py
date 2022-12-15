import numpy as np
from matplotlib import pyplot as plt

N_SAMPLES = int(5e+6)
BINS = np.linspace(-1, 8, 451)

x_mu = 5
y_mu = 3
scale_x = 5 * 0.05
scale_y = 3 * 0.05

ratio_samples = [
    x / y for x, y in zip(
        np.random.normal(x_mu, scale_x, size=N_SAMPLES), np.random.normal(y_mu, scale_y, size=N_SAMPLES)
    )
]

plt.hist(ratio_samples, bins=BINS, color='r', histtype='step')
plt.hist(np.random.normal(x_mu, scale_x, size=N_SAMPLES), bins=BINS, color='g', histtype='step')
plt.hist(np.random.normal(y_mu, scale_y, size=N_SAMPLES), bins=BINS, color='g', histtype='step')

new_ratio_samples = [
    x / y for x, y in zip(
        np.random.normal(x_mu, scale_x, size=N_SAMPLES), np.random.normal(y_mu, scale_y, size=N_SAMPLES)
    )
]

recovered_gaussian = [
    z * y for z, y in zip(new_ratio_samples, np.random.normal(y_mu, scale_y, size=N_SAMPLES))
]

plt.hist(recovered_gaussian, bins=BINS, color='b', histtype='step')

# recovered_gaussian = [ z * y_mu for z in new_ratio_samples ]

# plt.hist(recovered_gaussian, bins=BINS, color='y', histtype='step')

ratio_from_mean_samples = [ x / y_mu for x in np.random.normal(x_mu, scale_x, size=N_SAMPLES) ]

plt.hist(ratio_from_mean_samples, bins=BINS, color='k', histtype='step')

new_ratio_from_mean_samples = [ x / y_mu for x in np.random.normal(x_mu, scale_x, size=N_SAMPLES) ]

recovered_gaussian = [
    z * y for z, y in zip(new_ratio_from_mean_samples, np.random.normal(y_mu, scale_y, size=N_SAMPLES))
]

plt.hist(recovered_gaussian, bins=BINS, color='y', histtype='step')

# recovered_gaussian = [ z * y_mu for z in new_ratio_from_mean_samples ]

# plt.hist(recovered_gaussian, bins=BINS, color='y', histtype='step')

plt.show()

# y_mu = 5
# x_mu = 3
# scale_y = 5 * 0.05
# scale_x= 3 * 0.05

ratio_samples = [
    x / y for x, y in zip(
        np.random.normal(x_mu, scale_x, size=N_SAMPLES), np.random.normal(y_mu, scale_y, size=N_SAMPLES)
    )
]

ratio_samples_mean = np.mean(ratio_samples)

plt.hist(np.random.normal(x_mu, scale_x, size=N_SAMPLES), bins=BINS, color='g', histtype='step')
plt.hist(np.random.normal(y_mu, scale_y, size=N_SAMPLES), bins=BINS, color='g', histtype='step')

plt.hist(ratio_samples_mean * np.random.normal(y_mu, scale_y, size=N_SAMPLES), bins=BINS, color='r', histtype='step')

ratio_samples_proper_mean = np.mean(np.random.normal(x_mu, scale_x, size=N_SAMPLES)) * np.mean([ 1 / y for y in np.random.normal(y_mu, scale_y, size=N_SAMPLES) ])

plt.hist(ratio_samples_proper_mean * np.random.normal(y_mu, scale_y, size=N_SAMPLES), bins=BINS, color='b', histtype='step')

plt.show()

plt.hist(ratio_samples, bins=BINS, color='r', histtype='step')
plt.hist(np.random.normal(x_mu, scale_x, size=N_SAMPLES), bins=BINS, color='g', histtype='step')
plt.hist(np.random.normal(y_mu, scale_y, size=N_SAMPLES), bins=BINS, color='g', histtype='step')

new_ratio_samples = [
    x / y for x, y in zip(
        np.random.normal(x_mu, scale_x, size=N_SAMPLES), np.random.normal(y_mu, scale_y, size=N_SAMPLES)
    )
]

recovered_gaussian = [
    z * y for z, y in zip(new_ratio_samples, np.random.normal(y_mu, scale_y, size=N_SAMPLES))
]

plt.hist(recovered_gaussian, bins=BINS, color='b', histtype='step')

plt.show()



