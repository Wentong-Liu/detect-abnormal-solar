import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from utils import *


def fit_func(x):
    f = np.poly1d([6.45132019e-32, -2.47793836e-26, 3.76671693e-21, -2.85888380e-16,
                   1.12166857e-11, -2.13681764e-07, 1.72919800e-03, -3.60411423e+00,
                   1.45613808e-12])
    return f(x)


points = []

with open('data-processed.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        time = time2sec(row[1])
        solar = int(row[2])
        baseline = fit_func(time) * 0.6
        diff = max(0, baseline - solar)
        points.append([time, diff, solar])

points = np.asarray(points)

kmeans = KMeans(n_clusters=2)
kmeans.fit(points[:, [0, 1]])

y_km = kmeans.fit_predict(points[:, [0, 1]])

plot = 2  # change to 1 to plot fig.7L

plt.scatter(points[y_km == 0, 0], points[y_km == 0, plot], s=30, c='red')
plt.scatter(points[y_km == 1, 0], points[y_km == 1, plot], s=10, c='black')

plt.show()

# save labelled data in format: time, solar energy, normal or not
np.savetxt("data-labelled.csv", np.column_stack((points, y_km))[:, [0, 2, 3]], delimiter=",")
