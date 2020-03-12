'''
This program shows how The conjugate gradient  method is easily derived by examining
Ax=b equations in the linear system.
'''
import numpy as np
from numpy.linalg import *

xt = np.array([7.85971, 0.422926408, -0.073592239, -0.540643016, 0.010626163]) # This vector solves the system of linear equation
A = np.array([[0.2, 0.1, 1, 1, 0], [0.1, 4, -1, 1, -1], [1, -1, 60, 0, -2], [1, 1, 0, 8, 4], [0, -1, -2, 4, 700]])
b = np.array([1, 2, 3, 4, 5])
x = np.zeros(5)# Initial solution



r = b - np.dot(A, x) #Introducing the residual
p = r
rsold = np.dot(np.transpose(r), r)
i=0

while np.any(abs(x-xt)>=0.01):

        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 0:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        i+=1



print('The solution is',x)
print('Total no of iterations',i)








