import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import sys

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
        stateActionValues[State[0], State[1], Action] +=  LearningRate * (Reward +
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
NumWindyWorldEpisodes = 50

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

# Genetic Algorithm Parameters Learned Parameters
# LearningRate
# Epsilon


# Multiprocessing Parameters

thread_count = mp.cpu_count()
print('Thread count on this machine: ', thread_count, '\n')


# Genetic Algorithm Functions and Parameters

def CalculateFitness(individual):
    # Consider binary using Gray Encoding
    LRString, EpsString = '', ''
    for digit in individual[:7]: LRString += str(int(digit))
    for digit in individual[7:]: EpsString += str(int(digit))

    # Phenotypic transformation to values between 0.01 and 1.0
    LearningRate = round(graycodedict[LRString] / 127, 2) + 0.01
    Epsilon = round(graycodedict[EpsString] / 127, 2) + 0.01

    # Determine individual fitness
    totalsteps = 0
    for episode in range(NumWindyWorldEpisodes):
        steps = RunWindyWorld(LearningRate, Epsilon)
        totalsteps += steps
    return totalsteps, LearningRate, Epsilon

def crossover(parentlist):
    parent1 = parentlist[0]
    parent2 = parentlist[1]
    for child in range(2):
        loci = []
        if kpointcrossover == True:
            for j in range(knumcrossover): loci.append(np.random.randint(0, chromolen))
            loci.sort()

        if len(loci) == 0:
            pass
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

    return np.vstack((child1, child2))


islandpopsize = 14
islandgenerations = 4
numislands = 4
migrations = 3
chromolen = 14      # Representing Learning Rate and Epsilon between 0 and 127 binary values

print('Island Population Size: {}'.format(islandpopsize))
print('Island Generation: {}'.format(islandgenerations))
print('Number of Islands: {}'.format(numislands))
print('Number of Migrations: {}\n'.format(migrations))



#Randomly initialize population
Island1Pop = np.random.randint(2, size=(islandpopsize, chromolen))
Island2Pop = np.random.randint(2, size=(islandpopsize, chromolen))
Island3Pop = np.random.randint(2, size=(islandpopsize, chromolen))
Island4Pop = np.random.randint(2, size=(islandpopsize, chromolen))

bestindividual, bestfitness, bestLR, bestEps = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 100000, 0, 0
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

kpointcrossover = True
knumcrossover = 2


#Elitism
#Value of 1 or 2 accepted
numelite = 2


tournselect = True
tournsize = 3



def GeneticAlgorithm(Pop, bestindividual, bestfitness, bestLR, bestEps, islandNum):

    #Initialize Fitness Vector
    Fitness = np.zeros(islandpopsize)
    islandbestfitness = 10000

    for epoch in range(islandgenerations):

        sys.stdout.write('\rEvolving Island {} Generation {}'.format(islandNum, epoch+1))
        sys.stdout.flush()

        # Fitness Evaluation

        for individual in range(Pop.shape[0]):

            Fitness[individual], LearningRate, Epsilon = CalculateFitness(Pop[individual])

            if Fitness[individual] < islandbestfitness:
                islandbestindividual = Pop[individual]
                islandbestLR = round(LearningRate, 2)
                islandbestEps = round(Epsilon, 2)
                islandbestfitness = int(Fitness[individual])

            if Fitness[individual] < bestfitness:
                bestindividual = Pop[individual]
                bestLR = round(LearningRate, 2)
                bestEps = round(Epsilon, 2)
                bestfitness = int(Fitness[individual])

        bestfitnesslist.append(bestfitness)

        # Elitism
        if numelite == 1: NewPop = Pop[np.argsort(Fitness)][0]
        elif numelite == 2: NewPop = np.vstack((Pop[np.argsort(Fitness)][0], Pop[np.argsort(Fitness)][1]))

        while NewPop.shape[0] < islandpopsize:

            # Selection
            parentlist = []
            if tournselect == True:
                for parent in range(2):
                    tournlist, randlist = [], []
                    for k in range(tournsize):
                        RandNum = np.random.randint(0, islandpopsize)
                        randlist.append(RandNum)
                        tournlist.append(int(Fitness[RandNum]))
                    selectindividual = tournlist.index(min(tournlist))
                    parentlist.append(Pop[randlist[selectindividual]])

            # Crossover
            generationcrossover = False
            if np.random.random() < probcrossover:
                generationcrossover = True
                children = crossover(parentlist)

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

    print('\n\nIsland Least Number of Steps: ', islandbestfitness)
    print('Island Optimal Individual: ', islandbestindividual)
    print('Island Optimal Learning Rate: ', islandbestLR)
    print('Island Optimal Epsilon: ', islandbestEps, '\n')

    return Pop, islandbestindividual, bestindividual, bestfitness, bestLR, bestEps



#Required declaration for multiprocessing
if __name__ == "__main__":
    print('Beginning Island Model GA. In unlucky circumstances Windy World with high stochasticity can take a very '
          'long time to converge. \n')
    successfulmigrations = 1

    output = mp.Queue()


    for i in range(migrations):
        #Run GA for each Island and return the New Population and the best individual

        Processes = [mp.Process(target=GeneticAlgorithm,
                                args=(Island1Pop, bestindividual, bestfitness, bestLR, bestEps, 1)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island2Pop, bestindividual, bestfitness, bestLR, bestEps, 2)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island3Pop, bestindividual, bestfitness, bestLR, bestEps, 3)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island4Pop, bestindividual, bestfitness, bestLR, bestEps, 4))]

        for process in Processes:
            process.start()

        for process in Processes:
            process.join()

        results = [output.get() for process in Processes]

        print(results)

        # Island1Pop, Island1BestIndividual, bestindividual, bestfitness, bestLR, bestEps = \
        #     multiprocessing.Process(GeneticAlgorithm(Island1Pop, bestindividual, bestfitness, bestLR, bestEps, 1))
        # Island2Pop, Island2BestIndividual, bestindividual, bestfitness, bestLR, bestEps = \
        #     multiprocessing.Process(GeneticAlgorithm(Island2Pop, bestindividual, bestfitness, bestLR, bestEps, 2))
        # Island3Pop, Island3BestIndividual, bestindividual, bestfitness, bestLR, bestEps = \
        #     multiprocessing.Process(GeneticAlgorithm(Island3Pop, bestindividual, bestfitness, bestLR, bestEps, 3))
        # Island4Pop, Island4BestIndividual, bestindividual, bestfitness, bestLR, bestEps = \
        #     multiprocessing.Process(GeneticAlgorithm(Island4Pop, bestindividual, bestfitness, bestLR, bestEps, 4))


        BestIndividualList = [Island1BestIndividual, Island2BestIndividual, Island3BestIndividual, Island4BestIndividual]

        islandlist = []
        for i in range(numislands):
            parent1 = BestIndividualList[i]
            RandNum = np.random.randint(0, 4)
            while RandNum == i: RandNum = np.random.randint(0,4)
            parent2 = BestIndividualList[RandNum]

            children = crossover([parent1, parent2])

            migratoryinds = np.vstack((children, parent1))
            migratoryinds = np.vstack((migratoryinds, parent2))

            Fitness = np.zeros(4)

            for individual in range(migratoryinds.shape[0]):
                Fitness[individual], LearningRate, Epsilon = CalculateFitness(migratoryinds[individual])

            islandlist.append(migratoryinds[np.argsort(Fitness)[0]])
            islandlist.append(migratoryinds[np.argsort(Fitness)[1]])

        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[0]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[1]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[2]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[3]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[4]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[5]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[6]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[7]

        print('\nMigration {} Complete'.format(successfulmigrations))
        print('Overall Least Number of Steps: ', bestfitness)
        print('Overall Optimal Individual: ', bestindividual)
        print('Overall Optimal Learning Rate: ', bestLR)
        print('Overall Optimal Epsilon: ', bestEps, '\n\n')
        successfulmigrations += 1



print('List of Optimal Fitness: ', bestfitnesslist)


plt.figure()
plt.plot(bestfitnesslist)
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()





























