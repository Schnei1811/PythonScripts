import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


def RunWindyWorld(LearningRate, Epsilon):
    # Initialize steps, starting location, and first action initialized randomly
    episodesteps = 0
    State = startState
    Action = np.random.choice(PossibleActions)

    # keep going until get to the goal state
    while State != goal:
        newState = Locationmap[State[0]][State[1]][Action]

        # Act greedily with epsilon stochasticity
        if np.random.random() < Epsilon: newAction = np.random.choice(PossibleActions)
        else: newAction = np.argmax(stateActionValues[newState[0], newState[1], :])

        # Update State Action Value Table
        stateActionValues[State[0], State[1], Action] += LearningRate * (Reward +
                        stateActionValues[newState[0], newState[1], newAction] -
                        stateActionValues[State[0], State[1], Action])
        State, Action = newState, newAction
        episodesteps += 1
    return episodesteps

# Windy World Parameters

# Environmental Parameters
Height = 7
Width = 10
Wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
NumWindyWorldEpisodes = 1

# Agent Actions
NumActions = 4
Up, Down, Left, Right = 0, 1, 2, 3
PossibleActions = [Up, Down, Left, Right]

# Create location map for possible movement considering each action for every state
Locationmap = []
for h in range(0, Height):
    Locationmap.append([])
    for w in range(0, Width):
        dest = dict()
        dest[Up] = [max(h - 1 - Wind[w], 0), w]
        dest[Down] = [max(min(h + 1 - Wind[w], Height - 1), 0), w]
        dest[Left] = [max(h - Wind[w], 0), max(w - 1, 0)]
        dest[Right] = [max(h - Wind[w], 0), min(w + 1, Width - 1)]
        Locationmap[-1].append(dest)

# Initialize State Action Values
stateActionValues = np.zeros((Height, Width, NumActions))
startState = [np.random.randint(0, Height), 0]
goal = [3, 7]
Reward = -1

# Genetic Algorithm Parameters Hyperparameters
# LearningRate
# Epsilon




# Genetic Algorithm Parameters

popsize = 40
generations = 1
chromolen = 14      # Representing Learning Rate and Epsilon between 0 and 127 binary values

#Randomly initialize population
Pop = np.random.randint(2, size=(popsize, chromolen))
Fitness = np.zeros(popsize)
bestfitness, bestLearningRate, bestEpsilon = 100000, 0, 0
bestfitnesslist = []


#Create dictionary where key is graycode and value is the corresponding integer
graycodedict = {}
NUM_BITS = int(7)
for i in range(0, 1 << NUM_BITS):
    gray = i ^ (i >> 1)
    graycodedict["{0:0{1}b}".format(gray, NUM_BITS)] = i


#Mutation
fixedmutation = True
nummutations = 2

perlocimututation = False
perlocimututationprob = 0.3

swapmutation = False
swapmutationprob = 0.3


#Crossover
probcrossover = 0.5
crossnumchildren = 20

kpointcrossover = True
knumcrossover = 2

uniformcrossover = False
perlocicrossprob = 0.1


#Elitism
elitism = True
numelite = 2


tournselect = True
tournsize = 3


for epoch in range(generations):

    # Fitness Evaluation
    for individual in range(popsize):

        #Consider binary using Gray Encoding
        LRString, EpsString = '', ''
        for digit in Pop[individual][:7]: LRString += str(int(digit))
        for digit in Pop[individual][7:]: EpsString += str(int(digit))

        #Phenotypic transformation to values between 0.01 and 1.0
        LearningRate = round(graycodedict[LRString] / 127, 2) + 0.01
        Epsilon = round(graycodedict[EpsString] / 127, 2) + 0.01

        #Determine individual fitness
        totalsteps = 0
        for episode in range(NumWindyWorldEpisodes):
            steps = RunWindyWorld(LearningRate, Epsilon)
            totalsteps += steps
        print(totalsteps)
        Fitness[individual] = totalsteps
        if totalsteps < bestfitness:
            bestLearningRate = LearningRate
            bestEpsilon = Epsilon
            bestfitness = totalsteps

    bestfitnesslist.append(bestfitness)

    # Elitism
    if elitism == True:
        if numelite == 1:
            NewPop = Pop[np.argsort(Fitness)][0]
        elif numelite == 2:
            NewPop = np.vstack((Pop[np.argsort(Fitness)][0], Pop[np.argsort(Fitness)][1]))

    print(bestfitness)
    print(Fitness[np.argsort(Fitness)][0])


    while NewPop.shape[0] < popsize:

        # Selection
        parentlist = []
        if tournselect == True:
            for parent in range(2):
                tournlist, randlist = [], []
                for k in range(tournsize):
                    RandNum = np.random.randint(0, popsize)
                    randlist.append(RandNum)
                    tournlist.append(int(Fitness[RandNum]))
                selectindividual = tournlist.index(min(tournlist))
                parentlist.append(Pop[randlist[selectindividual]])


        # Crossover
        generationcrossover = False
        if np.random.random() < probcrossover:
            generationcrossover = True
            parent1 = parentlist[0]
            parent2 = parentlist[1]
            for child in range(2):
                loci = []
                if kpointcrossover == True:
                    for j in range(knumcrossover): loci.append(np.random.randint(0, chromolen))
                    loci.sort()


                if len(loci) == 0: pass
                elif len(loci) == 1:
                    child1 = np.append(parent1[0:loci[0]], parent2[loci[0]:])
                    child2 = np.append(parent2[0:loci[0]], parent1[loci[0]:])
                else:
                    for j in range(len(loci)):
                        if j == 0:
                            child1 = parent1[0:loci[j]]
                            child2 = parent2[0:loci[j]]
                        elif j % 2 == 0 and j == len(loci) - 1:
                            child1 = np.append(child1, parent1[loci[j - 1]:])
                            child2 = np.append(child2, parent2[loci[j - 1]:])
                        elif j % 2 == 1 and j == len(loci) - 1:
                            child1 = np.append(child1, parent2[loci[j - 1]:])
                            child2 = np.append(child2, parent1[loci[j - 1]:])
                        elif j % 2 == 0:
                            child1 = np.append(child1, parent1[loci[j - 1]:loci[j]])
                            child2 = np.append(child2, parent2[loci[j - 1]:loci[j]])
                        elif j % 2 == 1:
                            child1 = np.append(child1, parent2[loci[j - 1]:loci[j]])
                            child2 = np.append(child2, parent1[loci[j - 1]:loci[j]])

                    children = np.vstack((child1, child2))


        # Mutation
        if generationcrossover == True: mutantlist = [children[0], children[1]]
        else: mutantlist = [parentlist[0], parentlist[1]]

        for mutant in range(len(mutantlist)):
            Loci = []
            if fixedmutation == True:
                for k in range(nummutations):
                    Loci.append(np.random.randint(0, chromolen))
                for k in range(len(Loci)):
                    if mutantlist[mutant][Loci[k]] == 0: mutantlist[mutant][Loci[k]] = 1
                    else: mutantlist[mutant][Loci[k]] = 0

                NewPop = np.vstack((NewPop, mutantlist[0]))
        NewPop = np.vstack((NewPop, mutantlist[1]))

    Pop = NewPop
    print('Least Number of Steps: ', bestfitness)
    print('Optimal Learning Rate: ', bestLearningRate)
    print('Optimal Epsilon: ', bestEpsilon)




print('List of Optimal Fitness: ', bestfitnesslist)



























































plt.figure()
plt.plot(bestfitnesslist)
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()


