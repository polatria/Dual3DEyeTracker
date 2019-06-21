import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

eyeside = ['./Right', './Left']
eyepos = ['/elps', '/eybl']

trials = 0
while True:
    path = eyeside[0] + eyepos[0] + '/' + str(trials) + '.csv'
    if os.path.exists(path):
        trials += 1
    else:
        break

data = [[], []]  # 0: elps, 1: eybl

i = int(input('Which eye data do you use? (R:0/L:1)'))

for n in range(trials):
    for j in range(len(eyepos)):
        path = eyeside[i] + eyepos[j] + f'/{n}.csv'
        data[j].append(pd.read_csv(path, names=['x', 'y', 'z'], dtype=float))

fig = plt.figure()
ax = fig.gca(projection='3d')
cmap = plt.get_cmap('tab10')
for n in range(trials):
    j = 0
    ax.scatter(data[j][n]['x'], data[j][n]['y'], data[j][n]['z'])
ax.legend(frameon=False)
plt.show()
