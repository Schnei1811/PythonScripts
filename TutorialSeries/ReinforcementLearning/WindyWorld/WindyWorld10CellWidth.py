import pygame
import random
import numpy as np

class Player:
    def __init__(self, colour, xposition, yposition):
        self.colour = colour
        self.xposition = xposition
        self.yposition = yposition
        self.score = 0

pygame.init()

ingamemusicdict = {1: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/02 - Cloud Country.mp3',
                   2: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/04 - Settling In.mp3',
                   3: "F:/PythonDataBackUp/Agricola/StardewValleyOST/05 - Spring (It's A Big World Outside).mp3",
                   4: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/06 - Spring (The Valley Comes Alive).mp3',
                   5: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/14 - Summer (The Sun Can Bend An Orange Sky).mp3',
                   6: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/29 - Winter (Ancient).mp3',
                   7: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/56 - Mines (Crystal Bells).mp3',
                   8: "F:/PythonDataBackUp/Agricola/StardewValleyOST/22 - Fall (Raven's Descent).mp3",
                   9: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/08 - Pelican Town.mp3',
                   10: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/11 - Distant Banjo.mp3'}

ingamemusicorder = random.sample(range(1, 11), 10)

display_width = 1280
display_height = 800

numcellwidth = 10
cell_width = int(display_width / numcellwidth)
cell_height = int(display_height / numcellwidth)
if numcellwidth % 2 == 0: evendivider = cell_height / 2
else: evendivider = 0

#Discount Factor
Discount = .9

#State Transition Matrix
STM = [[0, 0.5, 0, 0.5, 0, 0, 0, 0, 0],
      [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0],
      [0, 0.5, 0, 0, 0, 0.5, 0, 0, 0],
      [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0.5, 0, 0, 0, 0, 0, 0.5],
      [0, 0, 0, 0.5, 0, 0, 0, 0.5, 0],
      [0, 0, 0, 0, 0.6, 0, 0.2, 0, 0.2],
      [0, 0, 0, 0, 0, 0.5, 0, 0.5, 0]]

#State Return Value
SRV = np.array([[-1], [-1], [-1], [-1], [50], [-1], [-1], [-1], [-1]]).astype(int)

initial_value_state = np.zeros((9, 1))

print((1 - np.multiply(Discount, STM))**-1 * SRV)
#print(SRV + np.multiply(Discount, STM))


# list of lists
adjLists = [[1, 2], [2, 3], [4], [4, 5], [5], []]

# testing
print("Neighbors of vertex 0: ", adjLists[0])
print("Neighbors of vertex 3: ", adjLists[3])

print("\nPrint all adjacency lists with corresponding vertex")
n = len(adjLists)
for v in range(0, n):
    print(v, ":", adjLists[v])


black = (0, 0, 0)
white = (255, 255, 255)
yellow = (232, 162, 0)
blue = (0, 242, 255)
darkblue = (36, 28, 237)
green = (76, 177, 34)
purple = (164, 73, 163)
red = (255, 0, 0)
bright_green = (0, 255, 0)

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Windy World')
gameIcon = pygame.image.load('WindyCloud.png')
pygame.display.set_icon(gameIcon)
clock = pygame.time.Clock()

def play_music():
    pygame.mixer.music.load(ingamemusicdict[ingamemusicorder[0]])
    for i, value in ingamemusicdict.items():
        pygame.mixer.music.queue(value)
    pygame.mixer.music.play(1)

def draw_borders():
    gameDisplay.fill(white)
    # Borders
    pygame.draw.rect(gameDisplay, blue, (cell_width*4, 0, cell_width*4, display_height))
    for i in range(0, numcellwidth + 1):
        pygame.draw.line(gameDisplay, black, (i * cell_width, 0), (i * cell_width, display_height))
        pygame.draw.line(gameDisplay, black, (0, i * cell_height), (display_width, i * cell_height))
    largeText = pygame.font.SysFont("comicsansms", 60)
    TextSurf, TextRect = text_objects("G", largeText)
    TextRect.center = (display_width - cell_width * 1.5, int((display_height / 2)+ evendivider) + cell_height)
    gameDisplay.blit(TextSurf, TextRect)

def draw_player(Agent):
    pygame.draw.circle(gameDisplay, Agent.colour, (Agent.xposition, Agent.yposition), 20)

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def game_loop():
    gameExit = False
    # play_music()
    if numcellwidth % 2 == 0: evendivider = cell_height / 2
    else: evendivider = 0
    Agent = Player(red, int(cell_width / 2), int((display_height / 2) + evendivider))
    KeyToggle = False

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        draw_borders()
        draw_player(Agent)

        if event.type == pygame.KEYDOWN and KeyToggle == False:
            if event.key == pygame.K_LEFT and Agent.xposition - cell_width > 0:
                Agent.xposition = Agent.xposition - cell_width
            if event.key == pygame.K_RIGHT and Agent.xposition + cell_width < display_width:
                Agent.xposition = Agent.xposition + cell_width
            if event.key == pygame.K_UP and Agent.yposition - cell_height > 0:
                Agent.yposition = Agent.yposition - cell_height
            if event.key == pygame.K_DOWN and Agent.yposition + cell_height < display_height:
                Agent.yposition = Agent.yposition + cell_height
            KeyToggle = True
        if event.type == pygame.KEYUP: KeyToggle = False


        pygame.display.update()
        clock.tick(60)
game_loop()