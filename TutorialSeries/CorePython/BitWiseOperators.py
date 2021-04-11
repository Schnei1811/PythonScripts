
#---- Binary AND ----

a = 50      #110010
b = 25      #011001
c = a & b   #010000     # and
d = a | b   #010000     # or
e = a ^ b   #010000     # not
print(c)
print(d)
print(e)

#---- Binary Right Shift ----

x = 240     # 11110000
y = x >> 2  # divide by itself twice
z = x << 2  # multiply by itself twice

print(y)
print(z)



print('{0:08b}'.format(6))

