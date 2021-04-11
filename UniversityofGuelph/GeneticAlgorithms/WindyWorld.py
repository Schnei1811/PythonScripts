import matplotlib.pyplot as plt
import numpy as np


# Environmental Parameters
Height = 7
Width = 10
Wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
NumEpisodes = 100

# Agent Actions
NumActions = 4
Up, Down, Left, Right = 0, 1, 2, 3
PossibleActions = [Up, Down, Left, Right]

# SARSA Hyperparameters
LearningRate = 0.01
Eps = 0.01
Reward = -1

# Initialize State Action Values
stateActionValues = np.zeros((Height, Width, NumActions))
startState = [np.random.randint(0, Height), 0]
goal = [3, 7]

# Create destination map for each action considering every state
Destinationmap = []
for h in range(0, Height):
    Destinationmap.append([])
    for w in range(0, Width):
        dest = dict()
        dest[Up] = [max(h - 1 - Wind[w], 0), w]
        dest[Down] = [max(min(h + 1 - Wind[w], Height - 1), 0), w]
        dest[Left] = [max(h - Wind[w], 0), max(w - 1, 0)]
        dest[Right] = [max(h - Wind[w], 0), min(w + 1, Width - 1)]
        Destinationmap[-1].append(dest)


def WindyWorld():
    # Initialize steps, starting location, and first action initialized randomly
    episodesteps = 0
    State = startState
    Action = np.random.choice(PossibleActions)

    # keep going until get to the goal state
    while State != goal:
        newState = Destinationmap[State[0]][State[1]][Action]
        # Act greedily with epsilon stochasticity
        if np.random.random() < Eps:
            newAction = np.random.choice(PossibleActions)
        else:
            newAction = np.argmax(stateActionValues[newState[0], newState[1], :])
        # Update State Action Value Table
        stateActionValues[State[0], State[1], Action] += \
            LearningRate * (Reward + stateActionValues[newState[0], newState[1], newAction] -
            stateActionValues[State[0], State[1], Action])
        State, Action = newState, newAction
        episodesteps += 1
    return episodesteps


PerEpisodeSteps = []
totalsteps = 0

for episode in range(NumEpisodes):
    steps = WindyWorld()
    totalsteps += steps
    PerEpisodeSteps.append(steps)

print(totalsteps)


plt.figure()
plt.plot(PerEpisodeSteps)
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()

