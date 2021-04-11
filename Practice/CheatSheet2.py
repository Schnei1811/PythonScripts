f = open('alice.txt', 'r')  #f = open('alice.txt','w')
outputstring = ''           #f.write(“Hello World”)
for line in f:
    outputstring += line
f.close()

import sys
sys.setrecursionlimit(10000)

grid = [[0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 2]]

def search(x,y):
    if grid[x][y] == 2:
        print('GOOAAAALL! at {}{}'.format(x,y))
        return True
    elif grid[x][y] == 1:
        return False
    elif grid[x][y] == 3:
        return False
    grid[x][y] = 3
    if ((x < len(grid)-1 and search(x+1,y))
        or (y > 0 and search(x, y-1))
        or (x > 0 and search(x-1, y))
        or (y < len(grid)-1 and search(x, y+1))):
        return True
    return False

class Person(object):
    def __init__(self, name):
        self.name = name

    def reveal_identity(self):
        print('My name is {}'.format(self.name))

class SuperHero(Person):
    def __int__(self, name, hero_name):
        super().__init__(name)
        self.hero_name = hero_name

    def reveal_identity(self):
        super().reveal_identity()
        print("... And I am {}".format(self.hero_name))

Stefan = Person("Stefan")
Stefan.reveal_identity()

wade = SuperHero("wade", 'deadpool')
wade.reveal_identity()