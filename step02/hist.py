import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
print(xs.shape)

plt.hist(xs, bins='auto', density=True)
plt.xlabel('Height')
plt.ylabel('Probability Density')
if not os.path.exists('output/step02-hist.png'):
    plt.savefig('output/step02-hist.png')

# 正規分布のパラメータは「サンプルの平均」と「サンプルの標準偏差」で
# 求めることができる
mu = np.mean(xs) # サンプルの平均
sigma = np.std(xs) # サンプルの標準偏差
print(f'mu: {mu}, sigma: {sigma}')

def normal(x, mu=0, sigma=1):
    # y = 1 / (np.sqrt(2*np.pi)*sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2*np.pi * sigma**2)

x = np.linspace(150, 190, 1000)
y = normal(x, mu, sigma)

plt.hist(xs, bins='auto', density=True, color='blue')
plt.plot(x, y, color='orange')
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
if not os.path.exists('output/step02-hist_comparison.png'):
    plt.savefig('output/step02-hist_comparison.png')