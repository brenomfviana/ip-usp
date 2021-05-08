import numpy as np

def FFT(f):
  N = len(f)
  if N <= 1:
    return f
  # Division
  even = FFT(f[0::2])
  odd = FFT(f[1::2])
  # Store combination of results
  temp = np.zeros(N).astype(np.complex64)
  # Only required to compute for half the frequencies
  # since u+N/2 can be obtained from the symmetry property
  for u in range(N//2):
    temp[u] = even[u] + np.exp(-2j * np.pi * u / N) * odd[u]
    temp[u+N//2] = even[u] - np.exp(-2j * np.pi * u / N) * odd[u]
  return temp

f = [0, 100, 200, 300]

print('f =', f)
print('fft =', FFT(f))
