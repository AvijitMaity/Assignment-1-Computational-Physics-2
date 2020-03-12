'''
In this program we will syudy how to write a Python code to apply the Power Method to find the dominant eigenvalue and
eigenvector of the matrix
'''
import numpy as np
from numpy.linalg import *

A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]) # Given matrix
B = eigvals(A) # Here we will get all 3 eigenvalues of the matrix
C = A
x = np.array([0, 1, 0])
y = np.array([1, 0, 0])

m = np.inner((np.dot(A, x)), y)
n= np.inner((np.dot(A, x)), y)
e_old=float(n/m)
i=0

while True:
    A = np.dot(A, C)
    n = np.inner((np.dot(A, x)), y)
    e_new = float(n/m)
    if abs(e_new-e_old)<=0.01:
        break
    else:
        e_old=e_new
    m=n
    i+=1

ev=np.dot(np.dot(A,np.transpose(C)),x) # eigen vector corresponding to the largest eigen value
eigen_vector=ev/np.linalg.norm(ev) # normalized eigen vector
print('The dominant eigenvalue using numpy.eigvals',np.amax(B))
print('The dominant eigenvalue',e_new)
print('Eigenvector corr the dominant eigenvalue ',eigen_vector)
print('Total no of iteration',i)


