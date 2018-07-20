import numpy as np

a = np.genfromtxt('POSCO.csv', delimiter=",")
b = np.genfromtxt('Hyundai_steel.csv', delimiter=",")

n = a[1:10] - b[1:10]
print(n)