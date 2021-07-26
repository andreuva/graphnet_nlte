import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm

directory = '../../data/prd_vlos/'
mode = 'train'
file = 'vlos'
with open(f'{directory}{mode}_{file}.pkl', 'rb') as filehandle:
    print(f'reading file: \t {directory}{mode}_{file}.pkl')
    data = pickle.load(filehandle)

data_means = np.array([])
for ind, point in enumerate(data):
    if len(point) > 200:
        data_means = np.append(data_means, np.mean(point))

# best fit to the histogram
(mu, sigma) = norm.fit(data_means)
print(mu, sigma)

# plot the histogram of the data
n, bins, patches = plt.hist(data_means, bins=100, alpha=0.6, density=True)
# plot the fitted pdf
yy = norm.pdf(bins, mu, sigma)
ll = plt.plot(bins, yy, 'r--', linewidth=2)

plt.show()
