'''
This program shows how The Jacobi method is easily derived by examining each of the
Ax=b equations in the linear system  in isolation.
'''
import numpy as np
from numpy import dot
A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,2],[1,1,0,8,4],[0,-1,-2,4,700]])
b = np.array([1,2,3,4,5])
x = np.array([0,0,0,0,0]) # Initial solution
xt=np.array([7.85971,0.422926408,-0.073592239,-0.540643016,0.010626163]) # This vector solves the system of linear equation

# Now we will decompose A into diagonal,the strictly lower-triangular, and the strictly upper-triangular parts of A

L =-np.tril(A,-1) # - L matrix represents the stricty lowaer triangular part of A
U =-np.triu(A,+1) # - U matrix represents the stricty upper triangular part of A
D=A+L+U # D matrix represents the digonal part of A

# Here we consider in each case, stop when the difference between the approximate solution
# vector and the true solution written above is less than 0.01

i=0
while np.any(abs(x-xt)>0.01) :  # tolerance of 0.01
    i+=1
    #In matrix terms, the definition of the Jacobi method in can be expressed as
    # x = D^-1(L+U)x + D^-1b
    x = dot(dot(np.linalg.inv(D),(L+U)),x)+ dot(np.linalg.inv(D),b)

print ('The solution is',x)
print('Total no of iterations',i) # It will count how many iterations we need  to reach this accuracy.
