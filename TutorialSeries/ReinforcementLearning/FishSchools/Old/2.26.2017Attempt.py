import pygame
import random
import numpy as np

class Animals:
    def __init__(self, colour, x_boundary, y_boundary, size_range=(4, 8), movement_range=(-1, 2)):
        self.size = random.randrange(size_range[0], size_range[1])
        self.colour = colour
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.x = random.randrange(0, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)
        self.movement_range = movement_range

    def move(self):
        self.move_x = random.randrange(self.movement_range[0], self.movement_range[1])
        self.move_y = random.randrange(self.movement_range[0], self.movement_range[1])
        self.x += self.move_x
        self.y += self.move_y

    def check_bounds(self):
        if self.x < 0: self.x = 0 + 25
        elif self.x > self.x_boundary: self.x = self.x_boundary - 25

        if self.y < 0: self.y = 0 + 25
        elif self.y > self.y_boundary: self.y = self.y_boundary - 25

    def __repr__(self):
        return 'Animal({}, {}, ({},{}))'.format(self.color, self.size, self.x, self.y)

    def __str__(self):
        return 'Animal of color:{}, size:{}, location: ({},{}))'.format(self.color, self.size, self.x, self.y)

class Fish(Animals):
    def __init__(self, color, x_boundary, y_boundary):
        Animals.__init__(self, color, x_boundary, y_boundary)
        self.color = COLOUR_FISH

    def move(self):
        self.x += random.randrange(-5, 6)
        self.y += random.randrange(-5, 6)

class Shark(Animals):
    def __init__(self, color, x_boundary, y_boundary):
        Animals.__init__(self, color, x_boundary, y_boundary)
        self.color = COLOUR_SHARK

    def move(self):
        self.x += random.randrange(-5, 6)
        self.y += random.randrange(-5, 6)
    #
    # def move(self, keypress):
    #     if keypress == 4: self.x += -5
    #     if keypress == 6: self.x += 5
    #     if keypress == 8: self.y += -5
    #     if keypress == 2: self.y += 5
    #     if keypress == 1:
    #         self.x += -5
    #         self.y += 5
    #     if keypress == 5:
    #         self.x += 0
    #         self.y += 0
    #     if keypress == 3:
    #         self.x += 5
    #         self.y += 5
    #     if keypress == 7:
    #         self.x += -5
    #         self.y += -5
    #     if keypress == 9:
    #         self.x += 5
    #         self.y += -5

def is_touching(b1, b2):
    return np.linalg.norm(np.array([b1.x, b1.y]) - np.array([b2.x, b2.y])) < (b1.size + b2.size)    #return true or false

def handle_collision(fish_list):
    blues, reds, greens = fish_list                     #list of dictionaries
    for fish_id, blue_blob, in blues.copy().items():     #copy list before iterating through it
        for other_blobs in blues, reds, greens:
            for other_blob_id, other_blob in other_blobs.copy().items():
                logging.debug('Checking if blobs are touching {} + {}'.format(repr(blue_blob), repr(other_blob)))
                if blue_blob == other_blob:             #blue blobs checking itself
                    pass
                else:
                    if is_touching(blue_blob, other_blob):
                        blue_blob + other_blob
                        if other_blob.size <= 0:
                            del other_blobs[other_blob_id]
                        if blue_blob.size <= 0:
                            del blues[blue_id]
    return blues, reds, greens

def draw_environment(fish_list):
    fish, shark = handle_collision(fish_list)
    print(fish_list)
    game_display.fill(COLOUR_OCEAN)
    for fish_dict in fish_list:
        for fish_id in fish_dict:
            fish = fish_dict[fish_id]
            pygame.draw.circle(game_display, fish.color, [fish.x, fish.y], fish.size)
            fish.move()
            fish.check_bounds()
    pygame.display.update()

def main():
    fish = dict(enumerate([Fish(COLOUR_FISH, WIDTH, HEIGHT) for i in range(STARTING_FISH)]))
    shark = dict(enumerate([Shark(COLOUR_SHARK, WIDTH, HEIGHT) for i in range(STARTING_SHARKS)]))
    playershark = shark[0]
    print(playershark)
    keypress = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP4: keypress = 4
                elif event.key == pygame.K_KP6: keypress = 6
                elif event.key == pygame.K_KP8: keypress = 8
                elif event.key == pygame.K_KP2: keypress = 2
                elif event.key == pygame.K_KP1: keypress = 1
                elif event.key == pygame.K_KP5: keypress = 5
                elif event.key == pygame.K_KP3: keypress = 3
                elif event.key == pygame.K_KP7: keypress = 7
                elif event.key == pygame.K_KP9: keypress = 9
            else: keypress = 5
        # playershark.move(keypress)
        # playershark.check_bounds()
        draw_environment([fish, shark])
        clock.tick(60)

STARTING_FISH = 3
STARTING_SHARKS = 1

WIDTH = 800
HEIGHT = 600
COLOUR_FISH = (236, 241, 48)
COLOUR_OCEAN = (24, 96, 228)
COLOUR_SHARK = (255, 0, 0)

game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ocean World")
clock = pygame.time.Clock()

if __name__ == '__main__':
    main()