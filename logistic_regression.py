import numpy as np
import matplotlib.pyplot as plt

def sigmoid(data):
    return np.where(data >= 0, 1 / (1 + np.exp(-data)), np.exp(data) / (1 + np.exp(data)))

def sigmoid_derivative(data):
    return sigmoid(data)*(1-sigmoid(data))

def CELoss(y, yHat):
    return -1 * np.sum(y * np.log(yHat))

def softmax(data):
    mx = np.max(data, axis = 1)
    mx = mx.reshape(mx.shape[0], 1)

    numerator = np.exp(data - mx)
    denominator = np.sum(numerator, axis = 1)
    denominator = denominator.reshape(denominator.shape[0], 1)

    return numerator / denominator

def LogisticRegression(X, y, li, lr):
    """
        Binarna logisticka regresija 
        input:
        X - tensor ulaznih podataka velicine (n, f) gde je n broj podataka, f broj featurea podatka
        y - tensor labela velicine (n, 1)
        li - broj iteracija ucenja
        lr - stopa ucenja

        out:
        L - niz gresaka za svaku iteraciju
        yHat - predvidjeni izlazi
        W - matrica tezina
        grafici greske i grafik y i yHat

        usage:
        >>> l, yh, w = LogisticRegression(X, y, 1e3, 1e-3)

        Klasifikacija:
        yHat < 0.5 -> Class 0
        yHat >= 0.5 -> Class 1

        Preko sigmoida stavlja izlazni logit velicine (1,1) u granicu izmedju 0 i 1
        Koristi Cross Entropy Loss za racunanje greske

        Normalizacija podataka pre koriscenja regresije je obavezna
    """

    # init w
    W = np.random.uniform(-10, 10, (X.shape[1], y.shape[1]))*0.1
    L = np.array([])

    # ucenje
    for i in range(int(li)):
        # forward pass
        z = X.dot(W)
        yHat = sigmoid(z)

        # dodavanje lossa
        L = np.append(L, CELoss(y, yHat))

        # backprop
        dyHat = -y / yHat + (1 - y) / (1 - yHat) # izvod CE-a
        dz = sigmoid_derivative(z) * dyHat
        dW = X.T.dot(dz)

        # update
        W -= dW*lr

        # print na svaku desetinu iteracija ucenja
        if i % int(li / 10) == 0:
            print("Iter {} od {}".format(i, int(li)))
            print("Loss {}".format(L[-1]))


    # plotovanje
    fig, ax = plt.subplots(2,1) # fig ti manje vise ne treba, ax su axis, subplots 2,1 je kao matrica grafika 2,1 velicine i mozes da indeksiras grafike (ako je npr (2,2) onda je [0][1] indeksiranje)
    ax[0].plot(L) # plotovanje lossa na [0] mestu
    ax[0].set_title("Loss")

    ax[1].plot(y, 'bo', yHat, 'ro') # plotovanje oba y na [1] mestu
    ax[1].set_title("yhat plot")

    return L, yHat, W

def MultivariateLogisticRegression(X, y, li, lr):
    """
        Multivariatna logisticka regresija za viseklasnu logisticku regresiju
        Moze da se koristi i za binarnu klasifikaciju ako se y predstavi kao (n, 2) tensor

        input:
        X - tensor ulaznih podataka velicine (n, f) gde je n broj podataka, f broj featurea
        y - tensor labela oblika (n, c) gde je c broj klasa i y je one hot encoded
        li - broj iteracija ucenja
        lr - stopa ucenja

        out:
        L - niz gresaka za svaku iteraciju ucenja
        yHat - tensor pretpostavljenih izlaza
        W - tensor naucenih tezina
        grafik greske kroz iteracije

        usage:
        >>> l, yh, w = MultivariateLogisticRegression(X, y, 1e3, 1e-3)

        Preko stabilnog softmaxa pravi c dimenzionalne logite sa verovatnocama za svaku klasu
        Preko Cross Entropy Lossa racuna gresku za svaku iteraciju

        Klasifikacija:
        indeks maksimalnog elementa vektora yHat je pretpostavljena klasa
        verovatnoca maksimalnog elementa vektora yHat je sigurnost pretpostavke

        Normalizacija ulaznih podataka je obavezna pre koriscenja regresije
    """

    # init w
    W = np.random.uniform(-10, 10, (X.shape[1], y.shape[1]))*0.1
    L = np.array([])

    # ucenje
    for i in range(int(li)):
        # forward
        z = X.dot(W)
        yHat = softmax(z)

        # loss racunanje
        L = np.append(L, CELoss(y, yHat))

        # backprop
        dz = yHat - y
        dW = X.T.dot(dz)

        # update
        W -= dW*lr

        # print na desetinu li
        if i % int(li / 10) == 0:
            print("Iter {} od {}".format(i, int(li)))
            print("Loss {}".format(L[-1]))

    # plotovanje
    plt.plot(L)
    plt.show()

    return L, yHat, W