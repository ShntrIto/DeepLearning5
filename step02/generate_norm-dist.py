import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)
samples = np.random.normal(mu, sigma, 10000)

# パラメータθを推定し，対象をモデル化できれば，モデルから新たなデータを生成することができることの簡単な例

plt.hist(xs, bins='auto', density=True, alpha=0.7, label='original')
plt.hist(samples, bins='auto', density=True, alpha=0.7, label='generated')
plt.xlabel('Height')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('output/step02-norm-dist.png')