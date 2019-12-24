import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
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
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


data = genfromtxt('data-labelled.csv', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(data[:, [0, 1]], data[:, [2]].ravel(), test_size=0.20)

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

# plot
plt.scatter(data[:, [0]], data[:, [1]], c=data[:, [2]], s=10, cmap='coolwarm')
plot_svc_decision_function(rbf_kernel)
plt.scatter(rbf_kernel.support_vectors_[:, 0], rbf_kernel.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none')
plt.show()
