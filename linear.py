import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(-5, 5, (1000, 1))
X = np.sort(X, axis = 0)

k_real = np.array([2]).reshape(1,1)
n_real = np.array([5]).reshape(1,1)

y = X @ k_real + n_real
#-------
def MSE(y, yh):
    return np.sum( (y - yh)**2 ) / (2*y.shape[0])

#-------
epochs = 100
alpha = 1e-1

L = []
#-------

def LinearRegression(X: np.ndarray, y: np.ndarray):

    k = np.random.uniform(-1, 1, (X.shape[1], y.shape[1]))
    n = np.random.uniform(-1, 1, (y.shape[1], 1))
    L = []

    for iter in range(epochs):
        if iter % (epochs / 10) == 0:
            print(f"iter {iter} od {epochs}")
            
        yh = X @ k + n

        L.append(MSE(y, yh))

        dyh = (yh - y) / X.shape[0]
        dk = np.transpose(X) @ dyh
        dn = np.sum(dyh, axis = 0)

        k -= alpha*dk
        n -= alpha*dn

    return L, k, n

L, k, n = LinearRegression(X, y)

f, a = plt.subplots(2, 1)
a[0].plot(L)
a[1].plot(X @ k_real + n_real, 'b-', X @ k + n, 'r-')
plt.show()
print(f"Real: k-{k_real}, n-{n_real}\nPredicted: k-{k}, n-{n}")