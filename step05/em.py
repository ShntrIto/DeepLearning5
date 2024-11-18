import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(path)
print(xs.shape)

# パラメータの初期化
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])

K = len(phis)
N = len(xs)
MAX_ITERS = 100
THRESHOLD = 1e-4

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2*np.pi)**d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y

def likelihood(xs, phis, mus, covs):
    # 対数尤度を計算する
    eps = 1e-8
    L = 0
    N = len(xs)
    
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N

current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E ステップ
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs) # 潜在変数を z を持つ任意の確率変数 q(z)
        
    # M ステップ
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        # phis
        phis[k] = qs_sum[k] / N
        
        # mus
        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]
        
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]
        
    print(f'{current_likelihood:.3f}') # 現在の対数尤度を確認
    
    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood) # 差分の絶対値
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood
    
# データの生成
N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    z = np.random.choice(K, p=phis)
    new_xs[n] = np.random.multivariate_normal(mus[z], covs[z])
    
# データの可視化
import matplotlib.pyplot as plt

x_range = np.arange(1.5, 6, 0.1)
y_range = np.arange(40, 100, 0.1)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = gmm(x, phis, mus, covs)
        
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Eruputions(mins)')
ax1.set_ylabel('Waiting(mins)')
ax1.contour(X, Y, Z, cmap='turbo')
ax1.scatter(xs[:, 0], xs[:, 1], c='blue', alpha=0.5, label='original')
ax1.scatter(new_xs[:, 0], new_xs[:, 1], c='orange', alpha=0.5, label='generated')
ax1.legend(['original', 'generated'])

plt.savefig('output/gmm_em.png')