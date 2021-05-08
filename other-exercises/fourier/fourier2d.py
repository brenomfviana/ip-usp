import numpy as np

def DFT2D(f):
  n, m = f.shape[0:2]
  F = np.zeros(f.shape, dtype=np.complex64)
  x = np.arange(n)
  y = np.arange(m)
  # For each frequency u=0..n-1
  for u in np.arange(n):
    # For each frequency v=0..m-1
    for v in np.arange(m):
      # For each image pixel
      F[u, v] += f[x, y] * np.exp(-1j * (2 * np.pi) * \
        ((u * x) / n + (v * y) / m))
