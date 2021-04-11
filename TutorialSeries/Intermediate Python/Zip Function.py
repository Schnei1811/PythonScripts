x = [1, 2, 3, 4]
y = [7, 6, 2, 1]
z = ['a', 'b', 'c', 'd']

for a,b in zip(x,y):
    print(a,b)

for a,b,c in zip(x,y,z):
    print(a,b,c)

print(zip(x,y,z))

for i in zip(x,y,z):
    print(i)

print(list(zip(x,y,z)))

print(dict(zip(y,z)))           #only works with two values

[print(a,b,c) for a,b,c in zip(x,y,z)]

[print(x,y) for x,y in zip(x,y)]        # variables not stored

for x,y in zip(x,y):                    # will rewrite x value
    print(x,y)

print(x)
