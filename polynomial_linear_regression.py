import numpy as np
import matplotlib.pyplot as plt
import random
#====

def Polynomial_Regression(X, y, degree, iterations, rate): # vraca parametre i loss
    """
    Radi regresiju nad polinomom tako sto iterativno azurira konstante uz stepene x

    input:  X ulaz x-eva u obliku np.arraya
            y vrednost polinoma za X-eve
            degree stepen polinoma
            iterations broj iteracija ucenja
            rate stopa ucenja

    output: naucene konstante uz stepene polinoma
            vrednosti polinoma za X i naucene konstante
            2x1 plot Lossa i naucenog i pravog poly

    usage:
    >>> X = np.array(range(-500, 500))*0.01
    >>> y = 3*X**3 - 2*X**2 + 3*X - 2
    >>> Polynomial_Regression(X, y, 3, 1e3, 1e-5)

    napomene:
    Nije bas brz
    Ako je mnogo datapointa jos vise se usporava
    Ako je y veliko (~1e5) nece raditi
    Rate treba da bude oko 1e-5
    Broj datapointsa u X direktno korelira sa brojem iteracija (inverzna zavisnost) (1e2 dp == 1e4 iter)
    """


    # varijable
    L = []
    a = [random.random()] # ovde ce mi biti konstante uz x i slobodni clan, njega prvog inicijalizujem
    datapoints = X.shape[0]

    for i in range(degree): # popunjavam ostale const
        a.append(random.random())
    #---

    while iterations > 0:
        if(iterations % 100 == 0):
            print(iterations)
        """ predict i loss """
        yHat = [] # prediction tj yHat mi je samo ta polinomska funkcija, ovo je najbolje sto sam u datom trenutku mogao da smislim za racunanje y-a polinomske funkcije
        for x in X:
            temp = 0
            for i in range(len(a)):
                temp += x**i * a[i] # pravim yHat tako sto samo uzimam x na stepem puta konstanta
            yHat.append(temp)

        temp = (y - yHat)**2 # formula MSE Lossa
        L.append(np.sum(temp) / (2*datapoints)) # dodajem MSE loss
        #---

        """ update params """
        da = []
        for i in range(len(a)):
            da.append( np.sum((yHat - y) * X**i) / datapoints ) # opsti slucaj za n-tu const: 1/n * suma (yHat - y) * x^n

        da = np.array(da)

        a -= da*rate

        iterations -= 1

    # plotovanje
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(L)
    ax[0].set_title("Loss")

    ax[1].plot(X, y, 'r', X, yHat, 'b')
    ax[1].set_title("Poly")

    return a, yHat