# Name: Breno Maurício de Freitas Viana
# NUSP: 11920060
# Course Code: SCC5830
# Year/Semester: 2021/1
# Assignment 3: Filtering


import math
import numpy as np
import imageio



# ----- (1) Read Parameters

# Get the location of the reference image `I`
imgref = input().rstrip()
# Get the filtering method identifier `F`
F = int(input())


# --- Read the filter parameters
size = 0
filter = None

if F == 1:
  size = int(input().rstrip())
  filter = np.fromstring(input().rstrip(), dtype=np.int32, sep=' ')
if F == 2:
  size = int(input().rstrip())
  filter = np.empty((size, size), dtype=np.int32)
  for i in range(size):
    filter[i] = np.fromstring(input().rstrip(), dtype=np.int32, sep=' ')
if F == 3:
  size = int(input().rstrip())


# --- Load image
I = imageio.imread(imgref)



# ----- (2) - Filtering

# --- Convolution functions

def conv_point1d(f, w, x, a):
  """
  Return the convolution of a 1D point.
  """
  # Get the subarray
  sub_f = f[x - a:x + a + 1]
  # Apply filter in the point x
  return np.sum(np.multiply(sub_f, w))

def conv_point2d(f, w, x, y, a, b):
  """
  Return the convolution of a 2D point.
  """
  # Create the sub 2d array
  sub_f = f[x - a:x + a + 1, y - b:y + b + 1]
  # Apply filter in the point x, y
  return np.sum(np.multiply(sub_f, w))

# Median filter

def median(f, x, y, a, b):
  """
  Return the median value of the `size`-neighbors of the given point.
  """
  # Create the sub 2d array
  sub_f = f[x - a:x + a + 1, y - b:y + b + 1]
  # Return the median
  arr = np.sort(np.asarray(sub_f).reshape(-1))
  return np.median(arr)


# --- Filters

def f1():
  """
  Filtering 1D.
  """
  # Get center of the filter
  c = int((size - 1) / 2)
  # Pad the flatten (1D array) image with wrapping
  If = np.pad(I.flatten(), (c), 'wrap')
  # Initialize the resulting image
  Ir = np.zeros(If.shape)
  # Apply 1D convulation in the image
  for x in range(c, Ir.shape[0] - c):
    Ir[x] = conv_point1d(If, filter, x, c)
  # Remove padding
  Ir = Ir[c:-c]
  # Return the resulting image with original shape
  return Ir.reshape(I.shape)

def f2():
  """
  Filtering 2D.
  """
  # Get center of the filter
  c = int((size - 1) / 2)
  # Pad the original image with edge
  Ip = np.pad(I, (c, c), 'edge')
  # Initialize the resulting image
  Ir = np.zeros(Ip.shape)
  N, M = Ir.shape
  for x in range(c, N - c):
    for y in range(c, M - c):
      Ir[x, y] = conv_point2d(Ip, filter, x, y, c, c)
  # Remove padding
  Ir = Ir[c:-c, c:-c]
  # Return the resulting image
  return Ir.reshape(I.shape)

def f3():
  """
  Median Filter.
  """
  a = int((size - 1) / 2)
  b = int((size - 1) / 2)
  # Pad the original image with constant
  Ip = np.pad(I, (size, size), 'constant', constant_values=(0, 0))
  # Initialize the resulting image
  Ir = np.zeros(Ip.shape)
  N, M = Ir.shape
  for x in range(size, N - size):
    for y in range(size, M - size):
      Ir[x, y] = median(Ip, x, y, a, b)
  # Remove padding
  Ir = Ir[size:-size, size:-size]
  # Return the resulting image
  return Ir.reshape(I.shape)


# --- Apply the given filter
def apply_filter():
  if F == 1:
    return f1()
  if F == 2:
    return f2()
  if F == 3:
    return f3()

# Apply filter
Ir = apply_filter()

# Normalize resultinrectg image and convert the values to uint8
Ir = (Ir - Ir.min()) / (Ir.max() - Ir.min())
Ir *= 255
Ir = Ir.astype(np.uint8)



# ----- (3) Comparing `Ir` against reference image `I`

# --- Calculate the root mean squared error
def rmse(I, Ir):
  N, M = I.shape
  rmse = 0.0
  for x in range(N):
    for y in range(M):
      rmse += math.pow(float(I[x, y]) - float(Ir[x, y]), 2)
  rmse = math.sqrt(rmse / (N * M))
  return round(rmse, 4)

# --- Print the computed result of RMSE
print(rmse(I, Ir))
