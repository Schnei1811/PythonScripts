

person = {'name': 'Jenn', 'age':23}

list = ['Jenn', 23]

sentence = 'My name is {0[name]} and I am {0[age]} years old'.format(person)
print(sentence)

sentence = 'My name is {name} and I am {age} years old'.format(name='Jenn', age='30')
print(sentence)

sentence = 'My name is {name} and I am {age} years old'.format(name='Jenn', age='30')
print(sentence)

sentence = 'My name is {name} and I am {age} years old'.format(**person)
print(sentence)

for i in range(1, 11):
    sentence = 'The value is {:02}'.format(i)
    print(sentence)

for i in range(1, 11):
    sentence = 'The value is {:03}'.format(i)
    print(sentence)

pi = 3.14159265

sentence = 'Pi is equal to {:.2f}'.format(pi)
print(sentence)

sentence = 'Pi is equal to {:.3f}'.format(pi)
print(sentence)

#comma separated and two decimal points
sentence = '1 MB is equal to {:,.2f} bytes'.format(1000**2)
print(sentence)

import datetime
#% signs coming from documentation
my_date = datetime.datetime(2016, 9, 24, 12, 30, 45)
print(my_date)


sentence = '{:%B %d, %Y}'.format(my_date)
print(sentence)

sentence = '{0:%B, %d, %Y} fell on a {0:%A} and was the {0:%j} day of the year'.format(my_date)
print(sentence)