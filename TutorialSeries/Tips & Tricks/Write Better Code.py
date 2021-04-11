cities = ['Marseille', 'Amsterdam', 'New York', 'London']

#bad way
# i = 0
# for city in cities:
#     print(i,city)
#     i += 1

#good way
for i, city in enumerate(cities):
    print(i, city)

x_list = [1,2,3]
y_list = [2,4,6]

#bad way
# for i in range(len(x_list)):
#     x = x_list[i]
#     y = y_list[i]
#     print(x, y)

#good way
for x,y in zip(x_list, y_list):
    print(x, y)

x = 10
y = -10
print('Before: x = {}, y = {}'.format(x,y))
#bad way
# tmp = y
# y = x
# x = tmp

#good way
x, y = y, x             #tuple unpacking
print('After: x = {}, y = {}'.format(x,y))


ages = {'Mary':31,'Jonathan':28}
#bad way
# if 'Dick' in ages:
#     age = ages['Dick']
# else:
#     age = 'Unknown'

#good way
age = ages.get('Dick','Unknown')
print('Dick is {} years old'.format(age))


needle = 'd'
haystack = ['a','b','c']
#bad way
# found = False
# for letter in haystack:
#     if needle == letter:
#         print('Found!')
#         found = True
#         break
# if not found:
#     print('Not Found!')

#good way
for letter in haystack:
    if needle == letter:
        print('Found!')
        break
else:   #If no break occurred
    print('Not Found!')


# The bad way
# f = open('example.txt')
# text = f.read()
# for line in text.split('\n'):
#     print(line)
# f.close()

#Good way
# f = open('example.txt')
# for line in f:
#     print(line)
# f.close()

# Best way                          #file opened and closed
with open('example.txt') as f:
    for line in f:
        print(line)

print('Converting!')
try:
    print(int('x'))
except:
    print('Conversion failed!')
else:   #If no-except
    print('Conversion successful!')
finally:    #Always
    print('Done!')