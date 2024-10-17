import os
import numpy as np
from scipy.stats import norm

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
mu = np.mean(xs)
sigma = np.std(xs)

p1 = norm.cdf(160, mu, sigma)
p2 = norm.cdf(170, mu, sigma)
print(f'P(X <= 160): {p1}')
print(f'P(X <= 170): {p2}')

p3 = norm.cdf(180, mu, sigma)
print(f'P(X > 180): {1- p3}')