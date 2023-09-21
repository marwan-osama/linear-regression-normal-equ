import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def w_ones_col(X):
    m = X.shape[0]
    ones_col = np.ones((m, 1))

    return np.concatenate((ones_col, X), axis=1)

def polynomial_trans(X, order):
    X_poly = np.copy(X)

    for i in range(2, order + 1):
        X_poly = np.concatenate((X_poly, X_poly ** i), axis=1)

    return w_ones_col(X_poly)

def compute_mse(X, y, w, order):
    m = X.shape[0]

    summation = np.sum((polynomial_trans(X, order) @ w - y) ** 2)

    return summation / (2 * m)

def normal_equation(X, y, order):
    X_poly = polynomial_trans(X, order)

    return np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

def plotting_data(x, w, order):
    X = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)

    Y = polynomial_trans(X, order) @ w

    return X, Y

def run_regression(X, y, axes, order):
    X_tr, X_cv, y_tr, y_cv = train_test_split(X, y, test_size=0.2, random_state=450)

    w = normal_equation(X_tr, y_tr, order)

    axes.scatter(X_tr, y_tr, label='TR')
    axes.scatter(X_cv, y_cv, label='CV')
    axes.autoscale(False)
    axes.plot(*plotting_data(X, w, order), c='r', label=f'order {order}')

    tr_mse = round(compute_mse(X_tr, y_tr, w, order), 2)
    cv_mse = round(compute_mse(X_cv, y_cv, w, order), 2)

    # axes.set_title(f' Training MSE = {tr_mse}\n CV MSE = {cv_mse}', y=-0.3, fontsize=8)
    axes.set_xlabel('size(sqft)')
    axes.set_ylabel('price x 1000')
    axes.legend()

    return np.array([tr_mse, cv_mse, order])

def load_house_data():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, './houses.txt')
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,0]
    y = data[:,4]

    return X.reshape(-1, 1), y.reshape(-1, 1)

X, Y = load_house_data()

mse_history = np.zeros((4, 3))

fig = plt.figure(figsize=(6, 8), layout="constrained")
spec = fig.add_gridspec(3, 2)

ax10 = fig.add_subplot(spec[1, 0])
mse_history[0, :] = run_regression(X, Y, ax10, 1)

ax11 = fig.add_subplot(spec[1, 1])
mse_history[1, :] = run_regression(X, Y, ax11, 2)

ax20 = fig.add_subplot(spec[2, 0])
mse_history[2, :] = run_regression(X, Y, ax20, 3)

ax21 = fig.add_subplot(spec[2, 1])
mse_history[3, :] = run_regression(X, Y, ax21, 4)

ax0 = fig.add_subplot(spec[0, :])
ax0.plot(mse_history[:, 2], mse_history[:, 1], label='C.V.')
ax0.plot(mse_history[:, 2], mse_history[:, 0], label='Tr')
ax0.set_xlabel('order')
ax0.set_ylabel('MSE')
ax0.legend()

fig.suptitle('Linear regression using the normal equation')
plt.show()
