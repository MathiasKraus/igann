import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from igann import IGANN

np.random.seed(0)


X = torch.from_numpy(np.random.rand(1000000, 10).astype(np.float32))
y = torch.from_numpy(np.random.rand(1000000).astype(np.float32))

n_samples = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
cpu_times = []
gpu_times = []

for n in n_samples:
  X, y = X.to('cpu'), y.to('cpu')
  print(n)
  m = IGANN(task='regression', device='cpu', n_estimators=100, early_stopping=100, n_hid=10, elm_alpha=5, boost_rate=1, interactions=0, verbose=0)
  start = time.time()
  m.fit(X[:n], y[:n])
  end = time.time()
  cpu_times.append(end - start)

  X, y = X.to('cpu'), y.to('cpu')
  m = IGANN(task='regression', device='cuda', n_estimators=100, early_stopping=100, n_hid=10, elm_alpha=5, boost_rate=1, interactions=0, verbose=0)
  start = time.time()
  m.fit(X[:n], y[:n])
  end = time.time()
  gpu_times.append(end - start)

plt.plot(np.log10(n_samples), cpu_times, label="cpu")
plt.plot(np.log10(n_samples), gpu_times, label="gpu")
plt.xlabel('Number of samples')
plt.ylabel('Training time (sec)')
plt.xticks(np.log10(n_samples), ["1e3", "5e3", "1e4", "5e4", "1e5", "5e5", "1e6"])
plt.legend()
plt.show()



np.random.seed(0)
X = torch.from_numpy(np.random.rand(100000, 100).astype(np.float32))
y = torch.from_numpy(np.random.rand(100000).astype(np.float32))

n_features = [2, 5, 10, 25, 50]
cpu_times = []
gpu_times = []

for n in n_features:
  X, y = X.to('cpu'), y.to('cpu')
  print(n)
  m = IGANN(task='regression', device='cpu', n_estimators=100, early_stopping=100, n_hid=10, elm_alpha=5, boost_rate=1, interactions=0, verbose=0)
  start = time.time()
  m.fit(X[:, :n], y)
  end = time.time()
  cpu_times.append(end - start)

  X, y = X.to('cpu'), y.to('cpu')
  m = IGANN(task='regression', device='cuda', n_estimators=100, early_stopping=100, n_hid=10, elm_alpha=5, boost_rate=1, interactions=0, verbose=0)
  start = time.time()
  m.fit(X[:, :n], y)
  end = time.time()
  gpu_times.append(end - start)

plt.plot(n_features, cpu_times, label="cpu")
plt.plot(n_features, gpu_times, label="gpu")
plt.xlabel('Number of features')
plt.ylabel('Training time (sec)')
plt.legend()
plt.show()