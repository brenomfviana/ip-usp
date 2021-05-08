import numpy as np
import matplotlib.pyplot as plt

# Discrete Fourier Transform (1D)
def DFT1D(f):
  n = f.shape[0]
  F = np.zeros(f.shape, dtype=np.complex64)
  # For each frequency u=0..n-1
  for u in np.arange(n):
    # For each element of `f` x=0..n-1
    for x in np.arange(n):
      F[u] += f[x] * np.exp(-1j * (2 * np.pi) * (u * x) / n)
  # Normalize the fourer
  return F / np.sqrt(n)

# Faster Discrete Fourier Transform (1D)
def fasterDFT1D(f):
  n = f.shape[0]
  F = np.zeros(f.shape, dtype=np.complex64)
  x = np.arange(n)
  # For each frequency u=0..n-1
  for u in np.arange(n):
    F[u] += f[x] * np.exp(-1j * (2 * np.pi) * (u * x) / n)
  # Normalize the fourer
  return F / np.sqrt(n)

# Define a signal
t = np.arange(0, 1, 0.005)
f = 1 * np.sin(t * (2 * np.pi) * 2) + 0.6 * np.cos(t * (2 * np.pi) * 8) + \
    0.4 * np.cos(t * (2 * np.pi) * 16)

# Computing DFT1D of `f`
F = DFT1D(f)

fq = np.arange(200)
plt.figure(figsize=(10, 4))

# Plot the magnitude of the DFT
plt.plot(fq, np.abs(F), 'r')
plt.savefig('dft_magnitude.png')

# Due to the symmetric property of the sine and cosine functions (which are
# similar in shape, but shifted), the Fourier Transform is also symmetric with
# respect to its central coefficient. One interpretation is that both the
# positive and negative frequency sinusoids are 90 degrees out of phase, but
# the magnitude of their response will be the same. In other words, they both
# respond to real signals in the same way.

# Plot only part of the frequencies
limit = 32

fq = np.arange(limit)
plt.figure(figsize=(10, 4))

# Plot the magnitude of the DFT
plt.plot(fq, np.abs(F[:limit]), 'r')
plt.savefig('dft_magnitude_limit.png')
