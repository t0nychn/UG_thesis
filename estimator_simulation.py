import random
import numpy as np
import matplotlib.pyplot as plt

def rand_ints():
    a = [random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a))]

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.legend()
    plt.show()

def rand_ints_short():
    a = [random.randrange(-100, 100) for i in range(20)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a))]

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.legend()
    plt.show()

def trending_rand_ints():
    a = [i+random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a))]

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.legend()
    plt.show()

def var_rand_ints():
    a = [i*random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a))]

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.legend()
    plt.show()

def anomaly_rand_ints():
    a = [20 if random.randrange(-10, 10)==5 else random.randrange(-10, 10) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a))]

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.legend()
    plt.show()