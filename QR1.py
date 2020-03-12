'''
This program shows the QR decomposition of
using numpy.linalg.qr
'''

import numpy as np
from numpy import linalg
A = np.array([[5,-2],[-2,8]])
Q,R = linalg.qr(A)
print('The orthogonal matrix is',Q)
print('The upper triangular matrix is',R)

# Now we will Use the decomposition to calculate the eigenvalues of the matrix

def check_diagonal(D):
    tolerence = 1.0e-8
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if (i != j):
                if (abs(D[i, j]) >= tolerence):
                    return False
            else:
                continue
    return True


D = A
while (check_diagonal(D) == False):
    Q = linalg.qr(D)[0]
    D = np.dot(np.transpose(Q), np.dot(D, Q))
print("the Eigen Values from QR  decomposition are: ")
for i in range(D.shape[0]):
    print(D[i, i])

print("Eigen Values calculated using linalg.eigh :")
print(linalg.eigh(A)[0])
