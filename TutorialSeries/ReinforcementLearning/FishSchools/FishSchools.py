import pygame
import numpy as np
import logging
import random
import time
from math import sqrt, fabs, atan2, degrees, pi
#np.set_printoptions(threshold=10000)

class Animal:
    def __init__(self, color, x_boundary, y_boundary):
        self.score = 0
        self.hunger = 100
        self.currentdirection = random.randrange(0, 7)
        self.movementcounter = 0
        self.rewardcounter = 0
        self.color = color
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.x = random.randrange(0, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)
        self.ecosystemdata = np.zeros(6)
        self.counter = 0

    def check_bounds(self):
        if self.x < 0: self.x = self.x_boundary
        elif self.x > self.x_boundary: self.x = 0

        if self.y < 0: self.y = self.y_boundary
        elif self.y > self.y_boundary: self.y = 0

    def __repr__(self):
        return 'Animal({}, {}, ({},{}))'.format(self.color, self.radius, self.x, self.y)

    def __str__(self):
        return 'Animal of location: ({},{})), colour: {}, score:{}'.format(self.x, self.y, self.color, self.score)

class Fish(Animal):

    def __init__(self, x_boundary, y_boundary):
        Animal.__init__(self, COLOUR_FISH, x_boundary, y_boundary)
        self.radius = 6
        self.perceiveforward = 50
        self.perceivewidth = 50
        self.perceiveradius = self.radius + 30
        self.index = 0
        self.Q = np.zeros([289, 8])
        self.state = 0
        self.action = random.randrange(0, 7)
        self.reward = 0
        self.learningrate = .85
        self.discount = .99

    def __add__(self, other_blob):
        if other_blob.color == COLOUR_SHARK:
            self.radius -= 50
            self.score = time.time() - starttime

    def perceive(self, other_animal):
        if other_animal.color == COLOUR_SHARK:
             distx = other_animal.x - self.x
             disty = other_animal.y - self.y
             disttot = int(sqrt(fabs(distx) + fabs(disty)))
             if (disttot & 1): disttot -= 1
             radshark = atan2(disty, distx)
             radshark %= 2 * pi
             degshark = int(5 * int(degrees(radshark)/5))
             if disttot == 4: self.index = (degshark / 5)
             if disttot == 6: self.index = (degshark / 5) + 72
             if disttot == 8: self.index = (degshark / 5) + 144
             if disttot == 10: self.index = (degshark / 5) + 216
        else: self.index = 0

    def QLearnAction(self):
        self.action = int(np.argmax(self.Q[self.state, :] + np.random.randn(1, 8) * (1./(self.index+1))))

    def QLearnUpdate(self):
        self.Q[self.state, self.action] += \
            self.learningrate * (self.reward + self.discount * np.max(self.Q[self.index, :]) - self.Q[self.state, self.action])

    def Respawn(self):
        print(self.index, np.count_nonzero(self.Q))
        self.reward = 0
        self.x = random.randrange(0, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)
        self.radius += 50

    def move(self):
        self.currentdirection = self.action
        if self.rewardcounter == 20:
            self.reward += 1
            self.rewardcounter = 0
        self.rewardcounter += 1
        if self.currentdirection == 0:
            self.headx = self.x
            self.heady = self.y - self.radius
            self.move_x = 0
            self.move_y = -3
        elif self.currentdirection == 1:
            self.headx = self.x + self.radius / 2
            self.heady = self.y - self.radius / 2
            self.move_x = 1
            self.move_y = -1
        elif self.currentdirection == 2:
            self.headx = self.x + self.radius
            self.heady = self.y
            self.move_x = 3
            self.move_y = 0
        elif self.currentdirection == 3:
            self.headx = self.x + self.radius / 2
            self.heady = self.y + self.radius / 2
            self.move_x = 1
            self.move_y = 1
        elif self.currentdirection == 4:
            self.headx = self.x
            self.heady = self.y + self.radius
            self.move_x = 0
            self.move_y = 3
        elif self.currentdirection == 5:
            self.headx = self.x - self.radius / 2
            self.heady = self.y + self.radius / 2
            self.move_x = -1
            self.move_y = 1
        elif self.currentdirection == 6:
            self.headx = self.x - self.radius
            self.heady = self.y
            self.move_x = -3
            self.move_y = 0
        elif self.currentdirection == 7:
            self.headx = self.x - self.radius / 2
            self.heady = self.y - self.radius / 2
            self.move_x = -1
            self.move_y = -1
        else:
            self.move_x = 0
            self.move_y = 0

        self.x += self.move_x
        self.y += self.move_y

class Shark(Animal):

    hungerscore = 100
    hungercounter = 0

    def __init__(self, x_boundary, y_boundary):
        Animal.__init__(self, COLOUR_SHARK, x_boundary, y_boundary)
        self.radius = 12
        self.perceiveforward = 60
        self.perceivewidth = 60
        self.perceiveradius = self.radius + 40

    def __add__(self, other_blob):
        if other_blob.color == COLOUR_SHARK:
            self.hunger += 50

    def hunger(self):
        if self.hungercounter == 10:
            self.hungerscore -= 1
            self.hungercounter = 0
        if self.hungerscore < 0: self.score = time.time() - starttime
        self.hungercounter += 1

    def move(self):
        if self.movementcounter == 10:
            self.currentdirection = self.currentdirection + random.randrange(-1, 2)
            self.movementcounter = 0
            self.score += 1
        self.movementcounter += 1
        if self.currentdirection == -1: self.currentdirection = 7
        elif self.currentdirection == 9: self.currentdirection = 0
        if self.currentdirection == 0:
            self.headx = self.x
            self.heady = self.y - self.radius
            self.move_x = 0
            self.move_y = -3
        elif self.currentdirection == 1:
            self.headx = self.x + self.radius / 2
            self.heady = self.y - self.radius / 2
            self.move_x = 1
            self.move_y = -1
        elif self.currentdirection == 2:
            self.headx = self.x + self.radius
            self.heady = self.y
            self.move_x = 3
            self.move_y = 0
        elif self.currentdirection == 3:
            self.headx = self.x + self.radius / 2
            self.heady = self.y + self.radius / 2
            self.move_x = 1
            self.move_y = 1
        elif self.currentdirection == 4:
            self.headx = self.x
            self.heady = self.y + self.radius
            self.move_x = 0
            self.move_y = 3
        elif self.currentdirection == 5:
            self.headx = self.x - self.radius / 2
            self.heady = self.y + self.radius / 2
            self.move_x = -1
            self.move_y = 1
        elif self.currentdirection == 6:
            self.headx = self.x - self.radius
            self.heady = self.y
            self.move_x = -3
            self.move_y = 0
        elif self.currentdirection == 7:
            self.headx = self.x - self.radius / 2
            self.heady = self.y - self.radius / 2
            self.move_x = -1
            self.move_y = -1
        else:
            self.move_x = 0
            self.move_y = 0

        self.x += self.move_x
        self.y += self.move_y

def is_perceived(b1, b2):
    return np.linalg.norm(np.array([b1.x, b1.y]) - np.array([b2.x, b2.y])) < (b1.perceiveradius + b2.perceiveradius)

def is_touching(b1, b2):
    return np.linalg.norm(np.array([b1.x, b1.y]) - np.array([b2.x, b2.y])) < (b1.radius + b2.radius)

def handle_collision(fish, sharks):
    for fish_id, fish_animal, in fish.copy().items():
        for other_animals in fish, sharks:
            for other_animals_id, other_animal in other_animals.copy().items():
                if is_touching(fish_animal, other_animal):
                    fish_animal + other_animal
                    other_animal + fish_animal
                    if fish_animal.radius <= 0:
                        Fish.Respawn(fish[fish_id])
                        #del fish[fish_id]

def draw_environment(fish, shark, animal_list):
    game_display.fill(COLOUR_OCEAN)
    handle_collision(fish, shark)
    for fish_id, fish_animal in fish.copy().items():
        for other_animals in fish, shark:
            for other_animals_id, other_animal in other_animals.copy().items():
                if is_perceived(fish_animal, other_animal):
                    Fish.perceive(fish[fish_id], other_animal)
        Fish.QLearnAction(fish[fish_id])
        Fish.QLearnUpdate(fish[fish_id])
        Fish.move(fish[fish_id])
    for shark_id, shark_animal in shark.copy().items():
        Shark.hunger(shark[shark_id])
        #if animalid.hungerscore < 0: del animalid
        Shark.move(shark[shark_id])
    for animal_dict in animal_list:
        for animal_id in animal_dict:
            animal = animal_dict[animal_id]
            pygame.draw.circle(game_display, animal.color, (animal.x, animal.y), animal.radius)
            pygame.draw.line(game_display, COLOUR_FIN, [animal.x, animal.y], [animal.headx, animal.heady], 2)
            animal.check_bounds()
    pygame.display.update()
    return

def main():
    fish = dict(enumerate([Fish(WIDTH, HEIGHT) for i in range(STARTING_FISH)]))
    sharks = dict(enumerate([Shark(WIDTH, HEIGHT) for i in range(STARTING_SHARKS)]))
    animal_list = [fish, sharks]
    while True:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            draw_environment(fish, sharks, animal_list)
            clock.tick(60)
        except Exception as e:
            logging.critical(str(e))
            pygame.quit()
            quit()
            break

starttime = time.time()
STARTING_FISH = 20
STARTING_SHARKS = 1

WIDTH = 300
HEIGHT = 200
COLOUR_FIN = (0, 0, 0)
COLOUR_OCEAN = (24, 96, 228)
COLOUR_FISH = (236, 241, 48)
COLOUR_SHARK = (255, 0, 0)

game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ocean World")
clock = pygame.time.Clock()

main()