import pprint
import numpy as np
a=np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
pprint.pprint(a)
'''
a=np.array([[1,2,3],
            [4,5,6]])
'''

print(a)
print(a.size)
print(a.shape)
print(a.dtype)
print(a[1][1][1])


b=np.zeros(4)
print(b)
c=np.ones(4)
print(c)
d=np.empty(10)
print(d)
e=np.eye(4)
f=np.identity(4, dtype=str)
print("identify")
print(e)
print(f)
print("range array")
r1=np.arange(8,81,8,dtype=int)
r2=np.arange(3,31,3,dtype=int)
print(r1)
print(r2)
result=r1%r2
print(result)