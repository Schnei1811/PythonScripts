# List comprehension will be faster but in memory. Generators will be 'slower' but will use less memory

#list comprehension
# xyz = [i for i in range(5)]
# xyz = []
# for i in range (5):
#     xyz.append(i)
#
# #generator
# xyz = (i for i in range(5))
# for i in xyz:
#     print(i)


input_list = [5,6,2,10,15,20,5,2,1,3]

def div_by_five(num):
    if num % 5 == 0:
        return True
    else:
        return False

xyz = (i for i in input_list if div_by_five(i))
# xyz = []                          # breakdown
# for i in input_list:
#     if div_by_five(i):
#         xyz.append(i)
print(xyz)

# for i in xyz:
#     print(i)
xyz = [i for i in input_list if div_by_five(i)]
print(xyz)

[[print(i,ii) for ii in range(5)] for i in range(5)]
# for i in range(5):
#   for ii in range(5):
#       print(i,ii)

xyz = [[(i,ii) for ii in range(5)] for i in range(5)]           #saved list
print(xyz)

#xyz = ([(i,ii) for ii in range(5)] for i in range(5))           #generator object
#print(xyz)
#for i in xyz:
#    print(i)

#xyz = (((i,ii) for ii in range(90000000000000000)) for i in range(900000000000))           #generator object

#for i in xyz:
#    for ii in i:
#        print(ii)

xyz = (print(i) for i in range(5))

for i in xyz:
    i