import numpy as np
import matplotlib.pyplot as plt
import scipy

def yj(x, l):
    return np.where(x < 0, -(np.pow(-x+1,2-l)-1)/(2-l), (np.pow(x+1,l)-1)/l)

def yj_inv(x, l):
    return np.where(x<0, -np.pow(((l-2)*x + 1), 1/(2-l))+1, np.pow((l*x+1),1/l)-1)

def plot_yj(samples=100000):
    model = lambda x : 1.9*(x - 1)

    rands = np.random.normal(size=samples)

    lambdas = np.linspace(0.01,1.99,1000)
    
    skews = np.array([scipy.stats.skew(yj(rands, l)) for l in lambdas])

    plt.plot(lambdas, skews, 'r--', label='Yeo-Johnson transform skews', zorder=5)
    plt.plot(lambdas, model(lambdas), 'b', label='Skew model', zorder=0)
    plt.title('Yeo-Johnson transform skews across choices of λ v Model')
    plt.xlabel('λ')
    plt.ylabel('Skew')
    plt.legend()

    plt.savefig(f'../plts/YJ/YJ_skew_modelling.png', dpi=300)
    plt.close()
