"""
Used for numerical computation of data
NumPy Is used rather than lists because it is FAST 
as it has fixed data type and uses vectorized method (compiling through C) instead of python loops. 
It uses contiguous memory meaning it is contiguous, no spaces in between
"""

import numpy as np
import numpy.linalg as linalg

a = np.array([1,2,3,4,5])
print(a)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

print(b.ndim)
print(b.shape)
print(a.dtype)
print(a.size)

print(b[1,1])
print(b[0,:])                   #first row all elements
print(b[:,2])                   #all rows, third column

print(np.zeros((2,3)))          #2 rows, 3 columns
print(np.ones((2,3)))           #2 rows, 3 columns
print(np.full((2,3), 5))        #2 rows, 3 columns filled with 5
print(np.random.rand(2,2))      #4 rows, 2 columns with random values
print(np.random.randint(0, 10, (2, 3)))     #2 rows, 3 columns with random integers between 0 and 10
print(np.identity(5))           #5x5 identity matrix

print(np.matmul(a, a))          #matrix multiplication
c =np.identity(3)
print(linalg.det(c))            #determinant of a matrix  
print(linalg.inv(c))            #inverse of a matrix
print(linalg.eig(c))            #eigenvalues and eigenvectors of a matrix

stats = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.min(stats, axis=1))    #minimum of each row
print(np.min(stats,axis=0))     #minimum of each column
print(np.max(stats))            #maximum of each row

d = np.array([[1,2,3],[4,5,6]])
print(d.reshape(3, 2))          #reshape to 3 rows, 2 columns