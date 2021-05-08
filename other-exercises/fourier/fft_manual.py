import numpy as np

N = 4
f = [0, 100, 200, 300]
print('f =', f)
print()

f_even = f[0::2]
print('f_even =', f_even)

f_even_even = f_even[0::2]
f_even_odd = f_even[1::2]
print('f_even_even =', f_even_even)
print('f_even_odd =', f_even_odd)

reseven0 = f_even_even[0] + np.exp(-2j * np.pi * 0 / N) * f_even_odd[0]
reseven1 = f_even_even[0] - np.exp(-2j * np.pi * 0 / N) * f_even_odd[0]
reseven = [reseven0, reseven1]
print('reseven =', reseven)
print()

f_odd = f[1::2]
print('f_odd =', f_odd)

f_odd_even = f_odd[0::2]
f_odd_odd = f_odd[1::2]
print('f_odd_even =', f_odd_even)
print('f_odd_odd =', f_odd_odd)

resodd0 = f_odd_even[0] + np.exp(-2j * np.pi * 0 / N) * f_odd_odd[0]
resodd1 = f_odd_even[0] - np.exp(-2j * np.pi * 0 / N) * f_odd_odd[0]
resodd = [resodd0, resodd1]
print('resodd =', resodd)

res0 = reseven[0] + np.exp(-2j * np.pi * 0 / N) * resodd[0]
res2 = reseven[0] - np.exp(-2j * np.pi * 0 / N) * resodd[0]

res1 = reseven[1] + np.exp(-2j * np.pi * 1 / N) * resodd[1]
res3 = reseven[1] - np.exp(-2j * np.pi * 1 / N) * resodd[1]

f_manual = np.array([res0, res1, res2, res3]).astype(np.complex64)
print(f_manual)
