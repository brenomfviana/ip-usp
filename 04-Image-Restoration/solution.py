# Name: Breno Maurício de Freitas Viana
# NUSP: 11920060
# Course Code: SCC5830
# Year/Semester: 2021/1
# Assignment 4: Image Restoration


import math
import numpy as np
import imageio
from scipy import fft



DENOISING = 1
CONSTRAINED = 2

# ----- (1) Read Parameters

# Get the location of the reference image `f`
f = input().rstrip()
# Get the location of the degraded image `g`
g = input().rstrip()
# Get the restoration filter identifier `F`
F = int(input().rstrip())
# Get the alpha for the restoration filters
alpha = float(input().rstrip())

# Get parameters of denoising filter
if F == DENOISING:
  # Get the coordinates of a flat rectangle in the image
  flat_rect = np.fromstring(input().rstrip(), dtype=np.int32, sep=' ')
  # Get the size of the filter n
  size = int(input().rstrip())
  # Get the denoising mode: "average" or "robust"
  mode = input().rstrip()
# Get parameters of constrained filter
elif F == CONSTRAINED:
  # Get the size of the gaussian filter
  k = int(input().rstrip())
  # Get sigma for the gaussian filter
  sigma = float(input().rstrip())


# --- Load images

# Reference image `f`
f = imageio.imread(f)
# Degraded image `g`
g = imageio.imread(g)



# ----- (2) - Restore Image

# --- Adaptive Denoising Filtering

# Denoising Modes

def average(g, x, y, a, b, dispn):
  """
  Return the average value of the given point.
  """
  # Create the sub 2d array
  sub_g = g[x - a:x + a + 1, y - b:y + b + 1]
  #
  # Calculate the noise dispersion
  displ = np.std(sub_g)
  # Fix the noise dispersion
  displ = dispn if displ == 0 else displ
  #
  # Calculate the centrality
  centr = np.average(sub_g)
  #
  # Return the result
  return g[x, y] - alpha * (dispn / displ) * (g[x, y] - centr)

def robust(g, x, y, a, b, dispn):
  """
  Return the robust restoration value of the given point.
  """
  # Create the sub 2d array
  sub_g = g[x - a:x + a + 1, y - b:y + b + 1]
  #
  # Calculate the noise dispersion
  arr = np.sort(np.asarray(sub_g).reshape(-1))
  q1 = np.percentile(arr, 25)
  q3 = np.percentile(arr, 75)
  displ = q3 - q1
  # Fix the noise dispersion
  displ = dispn if displ == 0 else displ
  #
  # Calculate the centrality
  centr = np.median(arr)
  #
  # Return the result
  return g[x, y] - alpha * (dispn / displ) * (g[x, y] - centr)


# Filter

def denoising(g):
  """
  Denoising filter.
  """
  a = int((size - 1) / 2)
  b = int((size - 1) / 2)
  # Pad the original image with symmetric
  gp = np.pad(g, (size, size), 'symmetric')
  #
  # Estimate the noise dispersion
  sub_g = gp[flat_rect[0]:flat_rect[1], flat_rect[2]:flat_rect[3]]
  dispn = 0
  if mode == 'average':
    dispn = np.std(sub_g)
  elif mode == 'robust':
    arr = np.sort(np.asarray(sub_g).reshape(-1))
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    dispn = q3 - q1
  # Fix the noise dispersion
  dispn = 1 if dispn == 0 else dispn
  #
  # Initialize the resulting image
  gr = np.zeros(gp.shape)
  N, M = gr.shape
  for x in range(size, N - size):
    for y in range(size, M - size):
      if mode == 'average':
        gr[x, y] = average(gp, x, y, a, b, dispn)
      elif mode == 'robust':
        gr[x, y] = robust(gp, x, y, a, b, dispn)
  # Remove padding
  gr = gr[size:-size, size:-size]
  #
  # Return the resulting image
  return gr.reshape(g.shape)


# --- Constrained Least Squares Filtering

# Auxiliary functions

def gaussian_filter(k=3, sigma=1.0):
  arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
  x, y = np.meshgrid(arx, arx)
  filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
  return filt / np.sum(filt)


# Filter

def constrained(g):
  """
  Constrained filter.
  """
  # Apply Fourier Transform
  gf = fft.rfft2(g)
  gN, _ = gf.shape
  #
  # Get gaussian filter
  H = gaussian_filter(k, sigma)
  HN, _ = H.shape
  a = gN // 2 - HN // 2
  H = np.pad(H, (a, a - 1), 'constant', constant_values=(0))
  H = fft.rfft2(H)
  #
  # Lapacian Operator
  P = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
  PN, _ = P.shape
  a = gN // 2 - PN // 2
  P = np.pad(P, (a, a - 1), 'constant', constant_values=(0))
  P = fft.rfft2(P)
  #
  # Calculate the resulting image
  gr = gf * (np.conj(H) / (abs(H * H) + alpha * abs(P * P)))
  gr = fft.irfft2(gr)
  gr = fft.fftshift(gr)
  #
  # Return the resulting image
  return gr


# --- Perform restoration

h = denoising(g) if F == DENOISING else constrained(g)
h = np.clip(h, 0, 255)
h = h.astype(np.uint8)

# imageio.imsave('result.png', h)



# ----- (3) Comparing `h` against reference image `f`

# --- Calculate the root mean squared error
def rmse(f, h):
  N, M = f.shape
  rmse = 0.0
  for x in range(N):
    for y in range(M):
      rmse += math.pow(float(h[x, y]) - float(f[x, y]), 2)
  rmse = math.sqrt(rmse / (N * M))
  return round(rmse, 4)

# --- Print the computed result of RMSE
print(rmse(f, h))
