import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.use("qt5cairo")
mpl.use('module://matplotlib-backend-sixel')

from csv import reader

import numpy as np

with open('records.csv', 'r') as f:
    data = list(map(lambda x: list(map(float, x)), (reader(f))))

data = np.array(data)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(*data.T)
plt.show()
