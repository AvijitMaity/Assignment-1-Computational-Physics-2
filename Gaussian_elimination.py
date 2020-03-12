'''
This program shows how to solve linear matrix equation using numpy.linalg.solve.
'''
import numpy as np
from numpy import linalg
A = np.array([[1,0.67,0.33], [0.45, 1, 0.55], [0.67,0.33, 1]])
b = np.array([2,2,2])
x = linalg.solve(A, b)
print('The solution is',x)

B=np.allclose(np.dot(A, x), b) # It will check the solution is correct or not
print(B)

'''
Note-
After solving this matrix equation in Gaussian elimination method we get solution vector x=(1,0.93,1).
Here we get solution vector using linalg.solve x= (1,1,1).
Both are more or less same.
'''
