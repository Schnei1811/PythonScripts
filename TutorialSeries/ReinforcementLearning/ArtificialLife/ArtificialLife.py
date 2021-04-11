import pygame
import random
import numpy as np
import time as tm

class Player:
    def __init__(self, xstate, ystate):
        self.xstate = xstate
        self.ystate = ystate
        self.xposition = int(cell_width * xstate + cell_width / 2)
        self.yposition = int(cell_height * ystate + cell_height / 2)
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
WORLD_HEIGHT = 15
# world width
WORLD_WIDTH = 15

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration
EPSILON = 0.2

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

# state action pair value
startState = [random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)]
waterState = [random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)]
foodState = [random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)]
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

cell_width = int(display_width / WORLD_WIDTH)
cell_height = int(display_height / WORLD_HEIGHT)
if WORLD_WIDTH % 2 == 0: evendivider = 0
else: evendivider = cell_height / 2

black = (0, 0, 0)
white = (255, 255, 255)
yellow = (232, 162, 0)
blue = (0, 242, 255)
darkblue = (0, 30, 255)
green = (76, 177, 34)
purple = (164, 73, 163)
red = (255, 0, 0)
bright_green = (0, 255, 0)

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Artificial Life')
gameIcon = pygame.image.load('ALIcon.jpg')
pygame.display.set_icon(gameIcon)
clock = pygame.time.Clock()

def play_music():
    pygame.mixer.music.load(ingamemusicdict[ingamemusicorder[0]])
    for i, value in ingamemusicdict.items(): pygame.mixer.music.queue(value)
    pygame.mixer.music.play(1)

def draw_borders():
    gameDisplay.fill(white)
    # Borders
    pygame.draw.rect(gameDisplay, darkblue, (cell_width * waterState[1], cell_height * waterState[0], cell_width, cell_height))
    pygame.draw.rect(gameDisplay, green, (cell_width * foodState[1], cell_height * foodState[0], cell_width, cell_height))
    for i in range(0, WORLD_WIDTH + 1):
        pygame.draw.line(gameDisplay, black, (i * cell_width, 0), (i * cell_width, display_height))
        pygame.draw.line(gameDisplay, black, (0, i * cell_height), (display_width, i * cell_height))
    largeText = pygame.font.SysFont("comicsansms", 50)
    TextSurf, TextRect = text_objects("W", largeText)
    TextRect.center = (cell_width * waterState[1] + cell_width / 2, cell_height * waterState[0] + cell_height / 2)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects("F", largeText)
    TextRect.center = (cell_width * foodState[1] + cell_width / 2, cell_height * foodState[0] + cell_height / 2)
    gameDisplay.blit(TextSurf, TextRect)

def draw_player(Agent):
    pygame.draw.circle(gameDisplay, red, (Agent.xposition, Agent.yposition), 20)

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def game_loop():
    gameExit = False
    #play_music()
    Agent = Player(startState[1], startState[0])

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if np.random.binomial(1, EPSILON) == 1:
            action = np.random.choice(actions)
            if action == 0: action = 'U'
            elif action == 1: action = 'D'
            elif action == 2: action = 'L'
            elif action == 3: action = 'R'
        else: action = optimalPolicy[Agent.ystate][Agent.xstate]

        #action = optimalPolicy[Agent.ystate][Agent.xstate]

        if action == 'R' and Agent.xposition + cell_width < display_width:
            Agent.xposition = Agent.xposition + cell_width
            Agent.xstate += 1
        elif action == 'L' and Agent.xposition - cell_width > 0:
            Agent.xposition = Agent.xposition - cell_width
            Agent.xstate -= 1
        elif action == 'U' and Agent.yposition - cell_height > 0:
            Agent.yposition = Agent.yposition - cell_height
            Agent.ystate -= 1
        elif action == 'D' and Agent.yposition + cell_height < display_height:
            Agent.yposition = Agent.yposition + cell_height
            Agent.ystate += 1
        if Agent.xstate == waterState[1] and Agent.ystate == waterState[0]:
            draw_player(Agent)
            print('WATER!')
            Agent.xstate, Agent.ystate = random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)
            Agent.xposition = int(cell_width * Agent.xstate + cell_width / 2)
            Agent.yposition = int(cell_height * Agent.ystate + cell_height / 2)
        if Agent.xstate == foodState[1] and Agent.ystate == foodState[0]:
            draw_player(Agent)
            print('FOOD!')
            Agent.xstate, Agent.ystate = random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)
            Agent.xposition = int(cell_width * Agent.xstate + cell_width / 2)
            Agent.yposition = int(cell_height * Agent.ystate + cell_height / 2)
            #foodState[0], foodState[1] = random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)
        if Agent.yposition < 0 or Agent.ystate < 0:
            Agent.yposition = int(0 + cell_height / 2)
            Agent.ystate = 0

        draw_borders()
        draw_player(Agent)
        pygame.display.update()
        clock.tick(60)
        tm.sleep(0.025)

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
    while currentState != foodState:
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
    #foodState[0], foodState[1] = random.randrange(0, WORLD_WIDTH, 1), random.randrange(0, WORLD_HEIGHT, 1)
    return time

# set up destinations for each action in each state
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destination = dict()
        destination[ACTION_UP] = [max(i - 1, 0), j]
        destination[ACTION_DOWN] = [max(min(i + 1, WORLD_HEIGHT - 1), 0), j]
        destination[ACTION_LEFT] = [max(i, 0), max(j - 1, 0)]
        destination[ACTION_RIGHT] = [max(i, 0), min(j + 1, WORLD_WIDTH - 1)]
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
        if [i, j] == foodState:
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
#print([str(w) for w in WIND])

game_loop()
