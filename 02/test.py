import numpy as np

d = {"a":1, "b":2, "c":3}
print(d["a"])
#print(d["a", "b"])
t = np.arange(12).reshape((3,4))
print("hello",t)
print(t[[1,2], [0,3]])

size = (5, 10)

(x, y) = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
print(x,y)


t = np.zeros(shape=(5, 6), dtype=object)
print(t)
t[[1, 3, 4], [0, 3, 4]] = None
print()
print(t)

t[4,5] = "A"
print(t[(0,0)])
print(type(t[0,0]))
print(type(t[4,5]))
print()
print(np.where(t=="A"))
print(np.where(t==0))
print(np.where(t==None))

#
# l = [["a", 0, None, 0], [0, 0, 0, None], [0,0,0,0]]
# print(l)
# t2 = np.array(l)
# print(type(t2[0,2]))