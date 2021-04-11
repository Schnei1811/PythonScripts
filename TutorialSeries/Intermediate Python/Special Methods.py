# x = (i for i in range(5))
# next(x)                 #moving iterator
# x.__next__()
# for i in x: print(i)
# dir(x)

class range_examp:

    def __init__(self, end, step=1):
        self.current = 0
        self.end = end
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        else:
            return_val = self.current
            self.current += self.step
            return return_val

x = range_examp(5)
x.__next__()
next(x)
for i in range_examp(5):  print(i)
dir(x)

def range_gen(end):
    current = 0
    while current < end:
        yield current
        current += 1

for i in range_gen(5): print(i)
x = range_gen(5)
x.__next__()
for i in x: print(i)
dir(range_gen)
dir(range_gen(5))
dir(range)
dir(range(5))
dir(i for i in range(5))

