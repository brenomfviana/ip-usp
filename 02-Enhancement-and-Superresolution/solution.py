# Name: Breno Maurício de Freitas Viana
# NUSP: 11920060
# Course Code: SCC5830
# Year/Semester: 2021/1
# Assignment 2: Enhancement and Superresolution


import math
import random
import numpy as np
import imageio



# ----- (1) Read Parameters

# Get the basename for the low resolution images `l_i`
lbn = input().rstrip()
# Get the high resolution image `H`
H = input().rstrip()
# Get the enhancement method identifier `F`
F = int(input())
# Get the enhancement method parameter `A`
A = float(input())



# ----- (2) Apply Enhancement Functions

# --- Load Images

# Load low resolution images
L = []
for i in [0, 1, 2, 3]:
  filename = lbn + str(i) + '.png'
  L.append(imageio.imread(filename))

# Load high resolution image
H = imageio.imread(H)


# --- Enhancement Transformations

LEVELS = 256

def hist_transf(hist, l):
  """
  Histogram transformation.
  """
  N, M = l.shape
  # Store the equalized image
  l_eq = np.zeros([N, M]).astype(np.uint8)
  # Apply histogram transformation
  for z in range(LEVELS):
    l_eq[np.where(l == z)] = ((LEVELS - 1) / float(N * M)) * hist[z]
  return l_eq

def e1(L):
  """
  Single-image Cumulative Histogram.
  """
  for i in [0, 1, 2, 3]:
    hist, _ = np.histogram(L[i], bins=LEVELS)
    hist = np.cumsum(hist)
    L[i] = hist_transf(hist, L[i])

def e2(L):
  """
  Joint Cumulative Histogram.
  """
  img = np.concatenate(L)
  hist, _ = np.histogram(img, bins=LEVELS)
  hist = np.cumsum(hist)
  hist = (hist - hist.min()) / (hist.max() - hist.min())
  N, M = L[0].shape
  hist *= N * M
  for i in [0, 1, 2, 3]:
    L[i] = hist_transf(hist, L[i])

def e3(L):
  """
  Gamma correction.
  """
  for l in L:
    N, M = l.shape
    for x in range(N):
      for y in range(M):
        l[x, y] = int(255 * math.pow(l[x, y] / 255.0, 1 / A))


# --- Apply Enhancement

ef = [e1, e2, e3]

if F in [1, 2, 3]:
  ef[F - 1](L)



# ----- (3) Apply Superresolution

N, M = L[0].shape

Hsr = np.zeros([N * 2, M * 2]).astype(np.uint8)


start = [(0, 0), (0, 1), (1, 0), (1, 1)]
for i in [0, 1, 2, 3]:
  sx, sy = start[i]
  for x in range(sx, N * 2, 2):
    for y in range(sy, M * 2, 2):
      Hsr[x, y] = L[i][int(x / 2), int(y / 2)]



# ----- (4) Comparing `Hsr` against reference image `H`

# --- Calculate the root mean squared error
rmse = 0.0
for x in range(N):
  for y in range(N):
    rmse += math.pow(float(H[x, y]) - float(Hsr[x, y]), 2)
rmse = math.sqrt(rmse / (N * M))

# --- Print the computed result of RSE
print(round(rmse, 4))
