import pygame
import random
import numpy as np
import matplotlib.pyplot as plt


class Player:
    def __init__(self, xstate, ystate):
        self.xstate = xstate
        self.ystate = ystate
        self.xposition = int(xstate + cell_width / 2)
        self.yposition = int(ystate + display_height / 2)
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

# world height
WORLD_HEIGHT = 7
# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

# state action pair value
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
startState = [3, 0]
goalState = [3, 7]
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

cell_width = int(display_width / WORLD_WIDTH)
cell_height = int(display_height / WORLD_HEIGHT)
if WORLD_WIDTH % 2 == 0: evendivider = 0
else: evendivider = cell_height / 2

black = (0, 0, 0)
white = (255, 255, 255)
yellow = (232, 162, 0)
blue = (0, 242, 255)
darkblue = (0, 0, 255)
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
    pygame.draw.rect(gameDisplay, blue, (cell_width * 3, 0, cell_width * 6, display_height))
    pygame.draw.rect(gameDisplay, darkblue, (cell_width * 6, 0, cell_width * 2, display_height))
    for i in range(0, WORLD_WIDTH + 1):
        pygame.draw.line(gameDisplay, black, (i * cell_width, 0), (i * cell_width, display_height))
        pygame.draw.line(gameDisplay, black, (0, i * cell_height), (display_width, i * cell_height))
    largeText = pygame.font.SysFont("comicsansms", 60)
    TextSurf, TextRect = text_objects("G", largeText)
    TextRect.center = (display_width - cell_width * 2.5, int((display_height / 2)+ evendivider))
    gameDisplay.blit(TextSurf, TextRect)

def draw_player(Agent):
    pygame.draw.circle(gameDisplay, red, (Agent.xposition, Agent.yposition), 20)

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def game_loop():
    gameExit = False
    play_music()
    Agent = Player(startState[1], startState[0])
    KeyToggle = False

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if event.type == pygame.KEYDOWN and KeyToggle == False:
            if event.key == pygame.K_LEFT and Agent.xposition - cell_width > 0:
                Agent.xposition = Agent.xposition - cell_width
                Agent.xstate -= 1
            if event.key == pygame.K_RIGHT and Agent.xposition + cell_width < display_width:
                Agent.xposition = Agent.xposition + cell_width
                Agent.xstate += 1
            if event.key == pygame.K_UP and Agent.yposition - cell_height > 0:
                Agent.yposition = Agent.yposition - cell_height
                Agent.ystate -= 1
            if event.key == pygame.K_DOWN and Agent.yposition + cell_height < display_height:
                Agent.yposition = Agent.yposition + cell_height
                Agent.ystate += 1
            if Agent.xstate == 7 and Agent.ystate == 3:
                draw_player(Agent)
                print('GOAL!')
                Agent.xstate, Agent.ystate = random.randrange(0, 5, 1), random.randrange(0, 7, 1)
                Agent.xposition = int(cell_width * Agent.xstate + cell_width / 2)
                Agent.yposition = int(cell_height * Agent.ystate + cell_height / 2)
            if Agent.xstate == 3 or Agent.xstate == 4 or Agent.xstate == 5 or Agent.xstate == 8:
                Agent.yposition = Agent.yposition - cell_height
                Agent.ystate -= 1
            if Agent.xstate == 6 or Agent.xstate == 7:
                Agent.yposition = Agent.yposition - cell_height * 2
                Agent.ystate -= 2
            if Agent.xstate == 7 and Agent.ystate == 3:
                draw_player(Agent)
                print('GOAL!')
                Agent.xstate, Agent.ystate = random.randrange(0, 5, 1), random.randrange(0, 7, 1)
                Agent.xposition = int(cell_width * Agent.xstate + cell_width / 2)
                Agent.yposition = int(cell_height * Agent.ystate + cell_height / 2)
            if Agent.yposition < 0:
                Agent.yposition = int(0 + cell_height / 2)
                Agent.ystate = 0
            KeyToggle = True

        if event.type == pygame.KEYUP: KeyToggle = False

        draw_borders()
        draw_player(Agent)
        pygame.display.update()
        clock.tick(60)

# play for an episode
def oneEpisode():
    # track the total time steps in this episode
    time = 0
    # initialize state
    currentState = startState
    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        currentAction = np.random.choice(actions)
    else:
        currentAction = np.argmax(stateActionValues[currentState[0], currentState[1], :])
    # keep going until get to the goal state
    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        if np.random.binomial(1, EPSILON) == 1:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(stateActionValues[newState[0], newState[1], :])
        # Sarsa update
        stateActionValues[currentState[0], currentState[1], currentAction] += \
            ALPHA * (REWARD + stateActionValues[newState[0], newState[1], newAction] -
            stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
        time += 1
    return time

# set up destinations for each action in each state
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destination = dict()
        destination[ACTION_UP] = [max(i - 1 - WIND[j], 0), j]
        destination[ACTION_DOWN] = [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
        destination[ACTION_LEFT] = [max(i - WIND[j], 0), max(j - 1, 0)]
        destination[ACTION_RIGHT] = [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
        actionDestination[-1].append(destination)


# play for 500 episodes to make sure to get a more converged policy
# episodeLimit = 200

# figure 6.4
episodeLimit = 10000
ep = 0
episodes = []
while ep < episodeLimit:
    time = oneEpisode()
    episodes.extend([ep] * time)
    ep += 1

# plt.figure()
# plt.plot(episodes)
# plt.xlabel('Time steps')
# plt.ylabel('Episodes')
# plt.show()

# display the optimal policy
optimalPolicy = []
for i in range(0, WORLD_HEIGHT):
    optimalPolicy.append([])
    for j in range(0, WORLD_WIDTH):
        if [i, j] == goalState:
            optimalPolicy[-1].append('G')
            continue
        bestAction = np.argmax(stateActionValues[i, j, :])
        if bestAction == ACTION_UP:
            optimalPolicy[-1].append('U')
        elif bestAction == ACTION_DOWN:
            optimalPolicy[-1].append('D')
        elif bestAction == ACTION_LEFT:
            optimalPolicy[-1].append('L')
        elif bestAction == ACTION_RIGHT:
            optimalPolicy[-1].append('R')
for row in optimalPolicy:
    print(row)
print([str(w) for w in WIND])

game_loop()
