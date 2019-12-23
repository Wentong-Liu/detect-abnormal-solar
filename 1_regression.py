import csv
import random

import numpy as np
from matplotlib import pylab as pen
from scipy.optimize import leastsq

from utils import *


def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, x, y):
    return fit_func(p, x) - y


def residuals_func_regularization(p, x, y):
    regularization = 0.001
    return np.append(residuals_func(p, x, y), regularization * np.sqrt(np.square(p)))  # L2范数作为正则化项


n = 9
# data points
point_x = []
point_y = []
# sunny curve
data_x_sunny = []
data_y_sunny = []
# non-sunny curve
data_x_other = []
data_y_other = []

weather_dict = {}
with open('data-weather.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        weather_dict[row[0]] = row[1]

# read data from csv file and prepare for approximation
with open('data-processed.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        time = time2sec(row[1])
        solar = int(row[2])
        if weather_dict[row[0]] == '1':
            data_x_sunny.append(time)
            data_y_sunny.append(solar)
        elif weather_dict[row[0]] == '2':
            data_x_other.append(time)
            data_y_other.append(solar)
        if random.randint(0, 100) < 1:  # plot 1% of all data points
            point_x.append(time)
            point_y.append(solar)

result_sunny = leastsq(residuals_func_regularization, np.random.randn(n), args=(data_x_sunny, data_y_sunny))
result_other = leastsq(residuals_func_regularization, np.random.randn(n), args=(data_x_other, data_y_other))

print('Fitting Parameters: ', result_sunny[0])
print('Fitting Parameters: ', result_other[0])

# plot
curve_x = np.linspace(0, 86400, 1000)
curve_y = np.linspace(0, 86400, 1000)
pen.plot(curve_x, fit_func(result_sunny[0], curve_x), 'r-', label='sunny')
pen.plot(curve_y, fit_func(result_other[0], curve_y), 'g-', label='non-sunny')
pen.plot(point_x, point_y, 'bo', label='data points')
pen.legend()
pen.show()
