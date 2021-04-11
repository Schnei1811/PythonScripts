

li = [9,1,8,2,7,3,6,4,5]

s_li = sorted(li)
print(s_li)
print(li)

li.sort()
print(li)

s_li = sorted(li, reverse=True)

li.sort(reverse=True)


tup = (9,1,8,2,7,3,6,4,5)

#tup.sort() error

s_tup = sorted(tup)

print(s_tup)   #now list

di = {'name':'Corey', 'job': 'Programming', 'age':None, 'os':'Mac'}

s_di = sorted(di)
print(di)


li = [-6,-5,-4,1,2,3]
s_li = sorted(li)
print(s_li)

s_li = sorted(li, key=abs)
print(s_li)

class Employee():
    def __init__(self, name, age, salary):
        self.name =name
        self.age = age
        self.salary = salary

    def __repr__(self):
        return '({},{},{})'.format(self.name, self.age, self.salary)

e1 = Employee('Carl', 37, 70000)
e2 = Employee('Sarah', 29, 80000)
e3 = Employee('John', 43, 90000)

employees = [e1, e2, e3]

def e_sort(emp):
    return emp.name

def age_sort(emp):
    return emp.age


s_employees = sorted(employees, key = e_sort)
print(s_employees)

s_employees = sorted(employees, key = age_sort)
print(s_employees)

s_employees = sorted(employees, key = age_sort, reverse=True)
print(s_employees)

s_employees = sorted(employees, key = lambda e: e.salary, reverse=True)
print(s_employees)

from operator import attrgetter

s_employees = sorted(employees, key = attrgetter('age'), reverse=True)
print(s_employees)





