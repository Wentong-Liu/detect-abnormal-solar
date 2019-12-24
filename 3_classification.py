import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from numpy import genfromtxt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


data = genfromtxt('data-labelled.csv', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(data[:, [0, 1, 2]], data[:, [3]].ravel(), test_size=0.20)

"""
Polynomial Kernel
"""
poly_kernel = SVC(kernel='poly', degree=8)
poly_kernel.fit(X_train, y_train)
poly_kernel.score(X_train, y_train)

y_pred = poly_kernel.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
RBF Kernel
"""

rbf_kernel = SVC(kernel='rbf')
rbf_kernel.fit(X_train, y_train)
rbf_kernel.score(X_train, y_train)

y_pred = rbf_kernel.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
Sigmoid Kernel
"""
sigmoid_kernel = SVC(kernel='sigmoid')
sigmoid_kernel.fit(X_train, y_train)
sigmoid_kernel.score(X_train, y_train)

y_pred = sigmoid_kernel.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot 2D
# plt.scatter(data[:, [0]], data[:, [2]], c=data[:, [3]], s=10, cmap='coolwarm')
# plot_svc_decision_function(rbf_kernel)
# plt.show()

"""
Linear SVM
"""
linear = SVC(kernel='linear')
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
Plot 3D
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = data[:, [0]]
ys = data[:, [2]]
zs = data[:, [1]]
label = data[:, [3]]
ax.scatter(xs[label == 0], ys[label == 0], zs[label == 0], marker='o')
ax.scatter(xs[label == 1], ys[label == 1], zs[label == 1], marker='^')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_zlabel('Diff')

plt.show()
