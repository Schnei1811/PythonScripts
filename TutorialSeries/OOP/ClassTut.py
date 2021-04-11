import datetime

class Employee:

    num_of_employees = 0
    raise_amount = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        Employee.num_of_employees += 1

    @property
    def email(self):
        return('{}.{}@company.com'.format(self.first, self.last))

    @property
    def fullname(self):
        return('{}.{}'.format(self.first, self.last))

    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)        #Employee.raise_amount works as well but does not allow instance changes

    @classmethod
    def set_raise_amount(cls, amount):
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True

    def __repr__(self):
        return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)

    def __str__(self):
        return '{} - {}'.format(self.fullname, self.email)

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return len(self.fullname)

class Developer(Employee):
    raise_amount = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        #Employee.__init__(self, first, last, pay)              Will also work but super preferred
        self.prog_lang = prog_lang

class Manager(Employee):

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None: self.employees = []
        else: self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname)


emp_1 = Employee('Stefan', 'Schneider', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

# print(Employee.num_of_employees)
# print(emp_1.pay)
# emp_1.apply_raise()
# print(emp_1.pay)
#
# emp_1.raise_amount = 1.05

Employee.set_raise_amount(1.05)

emp_str_1 = 'John-Doe-50000'
new_emp_1 = Employee.from_string(emp_str_1)
print(new_emp_1.email)

my_date = datetime.date(2016, 7, 10)
print(Employee.is_workday(my_date))

dev_1 = Developer('Guy', 'Stuff', 10000, 'Python')
dev_2 = Developer('Girl', 'Things', 10000, 'Java')

dev_1.apply_raise()
print(dev_1.pay)
print(dev_2.prog_lang)

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])
mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_1)

print(mgr_1.email)
mgr_1.print_emps()

print(isinstance(mgr_1, Manager))
print(isinstance(mgr_1, Employee))
print(isinstance(mgr_1, Developer))

print(issubclass(Manager, Employee))
print(issubclass(Manager, Developer))

print(emp_1.__repr__())
print(emp_1.__str__())

print(int.__add__(1, 2))
print(str.__add__('a', 'b'))
print('test'.__len__())
print(emp_1 + emp_2)
print(len(emp_1))

emp_1.fullname = 'Stefan Schneider'

print(emp_1.first)
print(emp_1.email)
print(emp_1.fullname)

del emp_1.fullname