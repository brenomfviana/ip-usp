import numpy as np


# Q3
f = np.matrix([[25,50,1,4],[255,52,2,5],[255,100,0,3],[255,100,3,120]])
w1 = np.matrix([[1,1,1],[1,-8,1],[1,1,1]])
w2 = np.matrix([[0,1/10,0],[1/10,6/10,1/10],[0,1/10,0]])

def conv_point(f, w, x, y):
  n,m = w.shape
  a = int((n - 1) / 2)
  b = int((m - 1) / 2)
  sub_f = f[x-a : x+a+1, y-b : y+b+1]
  w = np.flip(np.flip(w, 0), 1)
  value = np.sum(np.multiply(sub_f, w))
  return int(value)

print('w1 of g(1, 1) =', conv_point(f, w1, 1, 1))
print('w1 of g(2, 2) =', conv_point(f, w1, 2, 2))
print('w2 of g(1, 2) =', conv_point(f, w2, 1, 2))
print()


# Q4
wm = np.matrix([[1,1,1],[1,1,1],[1,1,1]])

def conv_point_median(f, w, x, y):
  n,m = w.shape
  a = int((n - 1) / 2)
  b = int((m - 1) / 2)
  sub_f = f[x-a : x+a+1, y-b : y+b+1]
  sub_f = np.delete(sub_f, [1,1])
  r = np.sort(np.asarray(sub_f).reshape(-1))
  return int(np.median(r))

def conv_point_max(f, w, x, y):
  n,m = w.shape
  a = int((n - 1) / 2)
  b = int((m - 1) / 2)
  sub_f = f[x-a : x+a+1, y-b : y+b+1]
  return int(np.max(sub_f))

def conv_point_midpoint(f, w, x, y):
  n,m = w.shape
  a = int((n - 1) / 2)
  b = int((m - 1) / 2)
  sub_f = f[x-a : x+a+1, y-b : y+b+1]
  max = np.max(sub_f)
  min = np.min(sub_f)
  return int((max + min) / 2)

print('median of g(1, 2) =', conv_point_median(f, wm, 1, 2))
print('max of g(2, 2) =', conv_point_max(f, wm, 2, 2))
print('midpoint of g(2, 1) =', conv_point_midpoint(f, wm, 2, 1))
