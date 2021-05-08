import numpy as np
import matplotlib.pyplot as plt
import imageio

# Number of observations
t = np.arange(0, 1, 0.001)
print("Number of observations: ", t.shape[0])

# Sine
mysine = np.sin(2 * np.pi * t)

# Sine * 4
mysine4 = np.sin(2 * np.pi * t * 4)

# Cosine * 4
mycos4 = np.cos(2 * np.pi * t * 4)

# Cosine * 18
mycos18 = np.cos(2 * np.pi * t * 18)

# Combining all the functions
myfun = mysine + mysine4 + mycos4 + mycos18
plt.figure(figsize=(10, 4))
plt.plot(t, myfun)
plt.savefig('myfun.png')

# Match sines and cosines with 4Hz (part of the signal)
Omega = 4
match_sin_4 = myfun * np.sin(Omega * (2 * np.pi) * t)
match_cos_4 = myfun * np.cos(Omega * (2 * np.pi) * t)
print('Sum of matching sines at 4Hz = %.2f' % np.sum(match_sin_4))
print('Sum of matching cosines at 4Hz = %.2f' % np.sum(match_cos_4))

# Match sines and cosines with 3Hz (not part of the signal)
Omega = 3
match_sin_3 = myfun * np.sin(Omega * (2 * np.pi) * t)
match_cos_3 = myfun * np.cos(Omega * (2 * np.pi) * t)
print('Sum of matching sines at 3Hz = %.4f' % np.sum(match_sin_3))
print('Sum of matching cosines at 3Hz = %.4f' % np.sum(match_cos_3))

# This procedure is able to detect that our function has 4Hz patterns while 3Hz
# is absent

# Plot the point-wise multiplication to observe this effect

plt.figure(figsize=(10, 4))
plt.plot(t, myfun, '-k')
plt.plot(t, match_sin_3, '--r')
plt.savefig('match_sin_3.png')

plt.figure(figsize=(10, 4))
plt.plot(t, myfun, '-k')
plt.plot(t, match_sin_4, '--r')
plt.savefig('match_sin_4.png')

def match_freqs(f, t, maxfreq):
  print("Coefficients of sine and cosine matching")
  for Omega in np.arange(0, maxfreq):
    match_sin = np.sum(f * np.sin(Omega * 2 * np.pi * t))
    match_cos = np.sum(f * np.cos(Omega * 2 * np.pi * t))
    print("%d\t%.1f\t%.1f" % (Omega, match_sin, match_cos))

match_freqs(myfun, t, 22)
