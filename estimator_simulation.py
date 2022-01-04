import random
import numpy as np
import matplotlib.pyplot as plt

def rand_ints():
    a = [random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(1, len(a)-1)]
    e.insert(0, np.NaN)

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.axhline(np.mean(a), label='actual mean', linestyle='--', color='blue', alpha=0.5)
    plt.axhline(np.mean(e), label='mean of estimates', linestyle='--', color='red', alpha=0.5)
    plt.legend()
    plt.show()

def rand_ints_short():
    a = [random.randrange(-100, 100) for i in range(20)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(1, len(a)-1)]
    e.insert(0, np.NaN)

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.axhline(np.mean(a), label='actual mean', linestyle='--', color='blue', alpha=0.5)
    plt.axhline(np.mean(e), label='mean of estimates', linestyle='--', color='red', alpha=0.5)
    plt.legend()
    plt.show()

def trending_rand_ints():
    a = [i+random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(len(a)-1)]
    e.insert(0, np.NaN)

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.axhline(np.mean(a), label='actual mean', linestyle='--', color='blue', alpha=0.5)
    plt.axhline(np.mean(e), label='mean of estimates', linestyle='--', color='red', alpha=0.5)
    plt.legend()
    plt.show()

def var_rand_ints():
    a = [i*random.randrange(-100, 100) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(1, len(a)-1)]
    e.insert(0, np.NaN)

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.axhline(np.mean(a), label='actual mean', linestyle='--', color='blue', alpha=0.5)
    plt.axhline(np.mean(e), label='mean of estimates', linestyle='--', color='red', alpha=0.5)
    plt.legend()
    plt.show()

def anomaly_rand_ints():
    a = [20 if random.randrange(-10, 10)==5 else random.randrange(-10, 10) for i in range(50)]
    e = [0.5*(np.mean(a[:i+1]) + np.mean(a[i:])) for i in range(1, len(a)-1)]
    e.insert(0, np.NaN)

    plt.plot(a, label='actual')
    plt.plot(e, label='estimate')
    plt.axhline(np.mean(a), label='actual mean', linestyle='--', color='blue', alpha=0.5)
    plt.axhline(np.mean(e), label='mean of estimates', linestyle='--', color='red', alpha=0.5)
    plt.legend()
    plt.show()