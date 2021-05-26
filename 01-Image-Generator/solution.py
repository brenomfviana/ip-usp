# Name: Breno Maurício de Freitas Viana
# NUSP: 11920060
# Course Code: SCC5830
# Year/Semester: 2021/1
# Assignment 1: Image Generator


import math
import random
import numpy as np



# ----- (1) Read parameters

# Get image filename for the reference image `r`
r = input().rstrip()
# Get lateral size of the synthesized image `C`
C = int(input())
# Get the function `F` to be used
F = int(input())
# Get parameter `Q`
Q = int(input())
# Get lateral size of the sampled image `N`
N = int(input())
# Get number of bits per pixel `B`
B = np.uint8(input())
# Get seed `S`
S = int(input())



# ----- (2) Generating Synthetic Image

# --- Auxiliary Variables and Funtions

# Initialize random algoritm generation with the seed `S`
random.seed(S)

# n-root function
def root(x, n):
  return x ** (1 / n)


# --- Functions

def f1(x, y):
  return x * y + 2 * y

def f2(x, y):
  return abs(math.cos(x / Q) + 2 * math.sin(y / Q))

def f3(x, y):
  return abs(3 * (x / Q) - root(y / Q, 3))

def f4(x, y):
  return random.random()

def f5(f):
  # Initialize the first position
  x, y = 0, 0
  f[x, y] = 1
  # Initialize auxiliary variables
  i = 0
  max = 1 + C * C
  # Perform random walk
  while i < max:
    # Get next step
    dx = random.randint(-1, 1)
    dy = random.randint(-1, 1)
    x = (x + dx) % C
    y = (y + dy) % C
    f[x, y] = 1
    i += 1

fc = [f1, f2, f3, f4]


# --- Create synthetic image

# Initialize default synthetic image `f`
f = np.zeros((C,C), dtype=float)

# Applying function `F`
if F in [1, 2, 3]:
  for x in range(C): # Row
    for y in range(C): # Col
      f[x, y] = fc[F - 1](x, y)
elif F == 4:
  # Calculates in raster order
  for x in range(C): # Row
    for y in range(C): # Col
      f[y, x] = f4(x, y)
else:
  # Random Walk
  f5(f)

# Normalizing the pixels' values with 16 bits (65536)
f = (f - f.min()) / (f.max() - f.min())
f *= 65535



# ----- (3) Sampling image from `f` to create `g`

# --- Downsampling

# Calculing the downsampling operator
step = int(C / N)
g = f[::step,::step]


# --- Quantizing

# Normalizing the pixels' values with 8 bits (256)
g = (g - g.min()) / (g.max() - g.min())
g *= 255

# Convert o usigned integer of 8 bits
g = g.astype(np.uint8)

# Apply bitwise operations
B = 8 - B
g = g >> B << B



# ----- (4) Comparing `g` against reference image `r`

# Load image
R = np.load(r)


# --- Calculate the root squared error
rse = 0.0
for x in range(N):
  for y in range(N):
    rse += math.pow(float(g[x, y]) - float(R[x, y]), 2)
rse = math.sqrt(rse)


# --- Print the computed result of RSE
print(round(rse, 4))
