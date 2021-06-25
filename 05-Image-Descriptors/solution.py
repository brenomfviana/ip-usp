# Name: Breno Maurício de Freitas Viana
# NUSP: 11920060
# Course Code: SCC5830
# Year/Semester: 2021/1
# Assignment 5: Image Descriptors


import math
import numpy as np
import imageio
from scipy import ndimage


np.seterr(divide='ignore', invalid='ignore')

LEVELS = 256

# ----- (1) Read Parameters

# Get the location of the object image `f`
f = input().rstrip()
# Get the location of the large image `g`
g = input().rstrip()
# Get the quantisation parameter `b`
b = int(input())


# --- Load images

# Object image `f`
f = imageio.imread(f)
# Large image `g`
g = imageio.imread(g)



# ----- (2) Preprocessing and Quantisation

def luminance(img):
  """
  Get a RGB image as input and return a black&white image.
  """
  N, M, _ = img.shape
  out = np.empty(img.shape)
  out = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
  return out.astype(np.uint8)


# --- Convert the images to black&white
f = luminance(f)
g = luminance(g)


# --- Quantise the images to `b` bits
B = 8 - b
f = f >> B
g = g >> B


# ----- (3) Image Descriptors

def nh_descriptor(f):
  """
  Return the normalized histogram descriptor.
  """
  hist, _ = np.histogram(f, bins=[i for i in range(2 ** b + 1)])
  hist = hist / hist.sum()
  dc = hist / np.linalg.norm(hist)
  return dc

def ht_descriptor(f):
  """
  Return the Haralick texture descriptors (intensity-level co-ocurrence matrix).
  """
  # Calculate the co-occurence matrix
  N, M = f.shape
  C = np.zeros((LEVELS, LEVELS))
  for x in range(N - 1):
    for y in range(M - 1):
      i = f[x, y]
      j = f[x + 1, y + 1]
      C[i][j] += 1
  C = C / C.sum()
  #
  # Computing the descriptors
  N, M = C.shape
  #
  energy = np.power(C, 2).sum()
  #
  epsilon = 0.001
  entropy = - (C * np.log(C + epsilon)).sum()
  #
  A = np.fromfunction(lambda i, j: (i - j) ** 2, (N, M), dtype=int)
  contrast = (1 / math.pow(N, 2)) * (C * A).sum()
  #
  mu_i, si_i = 0, 0
  mu_j, si_j = 0, 0
  for k in range(N):
    a1 = C[k,:].sum()
    mu_i += k * a1
    si_i += math.pow(k - mu_i, 2) * a1
    #
    a2 = C[:,k].sum()
    mu_j += k * a2
    si_j += math.pow(k - mu_j, 2) * a2
  #
  A = np.fromfunction(lambda i, j: (i - j) ** 2, (N, M), dtype=int)
  correlation = (A * C).sum() - mu_i * mu_j
  correlation /= (si_i * si_j)
  #
  homogeneity = 0
  #
  A = np.fromfunction(lambda i, j: (1 + abs(i - j)), (N, M), dtype=int)
  homogeneity = (C * A).sum()
  #
  # Return the Haralick texture descriptors
  dt = np.array([energy, entropy, contrast, correlation, homogeneity])
  dt = dt / np.linalg.norm(dt)
  return dt

def hg_descriptor(f):
  """
  Return the histogram of oriented gradients descriptor.
  """
  wsx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  wsy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  #
  f = f.astype(np.float64)
  fx = ndimage.convolve(f, wsx)
  fy = ndimage.convolve(f, wsy)
  #
  N, M = f.shape
  #
  div = np.sqrt(np.power(fx, 2) + np.power(fy, 2)).sum()
  Mg = np.sqrt(np.power(fx, 2) + np.power(fy, 2)) / div
  #
  sigma = np.zeros(f.shape)
  sigma = np.arctan(fy / fx) + np.pi / 2
  sigma = np.degrees(sigma)
  sigma = np.digitize(sigma, np.arange(0, 180, 20))
  sigma = sigma.astype(np.uint8)
  #
  dg = np.zeros(9)
  for x in range(N):
    for y in range(M):
      dg[sigma[x][y] - 1] += Mg[x][y]
  #
  dg = dg / np.linalg.norm(dg)
  return dg


# --- Compute the image descriptors

# Calculate the object image descriptors
dc = nh_descriptor(f)
dt = ht_descriptor(f)
dg = hg_descriptor(f)

d = np.concatenate((dc, dt, dg))



# ----- (4) Finding Our Object

def distance(d, di):
  """
  Calculate the distance of two descriptors.
  """
  return math.sqrt(np.power(d - di, 2).sum())


# --- Search for the object image location in the original image

size = f.shape[0]
step = size // 2
N, M = g.shape
N = N // step
M = M // step

dist = np.iinfo(np.uint8).max

pos_x = None
pos_y = None

for i in range(N - 1):
  for j in range(M - 1):
    # Calculate the window
    window = g[i*step:i*step+size, j*step:j*step+size]
    # Calculate the descriptors of the window
    window_dc = nh_descriptor(window)
    window_dt = ht_descriptor(window)
    window_dg = hg_descriptor(window)
    window_d = np.concatenate((window_dc, window_dt, window_dg))
    # Calculate the distance between the window and the object image
    ndist = distance(d, window_d)
    if dist > ndist:
      dist = ndist
      pos_x, pos_y = i, j


# --- Print the found location

print(pos_x, pos_y)
