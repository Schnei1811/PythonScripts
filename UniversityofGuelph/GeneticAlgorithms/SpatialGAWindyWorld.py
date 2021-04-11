import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def RunWindyWorld(LearningRate, Epsilon):

    #Windy World Parameters

    Height = 7
    Width = 10
    Wind = [0, 0, 0, 1, 1, np.random.randint(0,1),
            np.random.randint(0, 2), np.random.randint(0,2), np.random.randint(0,1), 0]

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



# Genetic Algorithm Functions and Parameters

def CalculateFitness(individual, graycodedict):

    # Consider binary using Gray Encoding
    LRString, EpsString = '', ''
    for digit in individual[:7]: LRString += str(int(digit))
    for digit in individual[7:]: EpsString += str(int(digit))

    # Phenotypic transformation to values between 0.01 and 1.0
    LearningRate = round(graycodedict[LRString] / 127, 2) + 0.01
    Epsilon = round(graycodedict[EpsString] / 127, 2)

    # Determine individual fitness considering the total number of steps taken to solve 1xdif-50 unique episodes of
    # Windy World. The fewer the number of steps, the better the parameters and higher the fitness

    totalsteps = 0
    NumWindyWorldEpisodes = 50
    for episode in range(NumWindyWorldEpisodes):
        steps = RunWindyWorld(LearningRate, Epsilon)
        totalsteps += steps
    return totalsteps, LearningRate, Epsilon



def crossover(parentlist):

    # 2-point crossover
    knumcrossover, chromolen = 2, 14

    parent1 = parentlist[0]
    parent2 = parentlist[1]
    for child in range(2):
        loci = []
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




def GeneticAlgorithm(Pop, bestindividual, bestfitness, bestLR, bestEps, islandNum, graycodedict, return_dict):

    chromolen = 14  # Representing Learning Rate and Epsilon between 0 and 127 binary values
    islandpopsize = 20
    islandgenerations = 5
    Fitness = np.zeros(islandpopsize)
    islandbestfitness = 1000000

    # Fixed Mutation
    nummutations = 2

    # K-point Crossover (K = 2)
    probcrossover = 0.5

    # Elitism
    numelite = 1

    # Tournament Selection
    tournsize = 3

    for epoch in range(islandgenerations):
        print('Evolving Island {} Generation {}'.format(islandNum, epoch+1))

        # For each individual calculate fitness and compare to present best island values
        for individual in range(islandpopsize):
            Fitness[individual], LearningRate, Epsilon = CalculateFitness(Pop[individual], graycodedict)
            if Fitness[individual] < islandbestfitness:
                islandbestindividual = Pop[individual]
                islandbestLR = round(LearningRate, 2)
                islandbestEps = round(Epsilon, 2)
                islandbestfitness = int(Fitness[individual])

        # Elitism
        if numelite == 1: NewPop = Pop[np.argsort(Fitness)][0]
        elif numelite == 2: NewPop = np.vstack((Pop[np.argsort(Fitness)][0], Pop[np.argsort(Fitness)][1]))

        while NewPop.shape[0] < islandpopsize:

            # 3 Way Tournament Selection
            parentlist = []
            for parent in range(2):
                tournlist, randlist = [], []
                for k in range(tournsize):
                    RandNum = np.random.randint(0, islandpopsize)
                    randlist.append(RandNum)
                    tournlist.append(int(Fitness[RandNum]))
                selectindividual = tournlist.index(min(tournlist))
                parentlist.append(Pop[randlist[selectindividual]])

            # 2-Point Crossover
            generationcrossover = False
            if np.random.random() < probcrossover:
                generationcrossover = True
                children = crossover(parentlist)

            if generationcrossover == True: mutantlist = [children[0], children[1]]
            else: mutantlist = [parentlist[0], parentlist[1]]

            # 2 Valued Fixed Point Mutation
            for mutant in range(len(mutantlist)):
                Loci = []
                for k in range(nummutations):
                    Loci.append(np.random.randint(0, chromolen))
                for k in range(len(Loci)):
                    if mutantlist[mutant][Loci[k]] == 0: mutantlist[mutant][Loci[k]] = 1
                    else: mutantlist[mutant][Loci[k]] = 0

                    NewPop = np.vstack((NewPop, mutantlist[0]))
                    NewPop = np.vstack((NewPop, mutantlist[1]))

        Pop = NewPop

    print('\n\nSummary for Island: ', islandNum)
    print('Island Least Number of Steps: ', islandbestfitness)
    print('Island Optimal Individual: ', islandbestindividual)
    print('Island Optimal Learning Rate: ', islandbestLR)
    print('Island Optimal Epsilon: ', islandbestEps, '\n')

    #Return shared dictionary rather than traditional return required for multiprocessing
    return_dict[islandNum] = Pop, islandbestindividual, islandbestLR, islandbestEps, islandbestfitness



if __name__ == "__main__":

    # Island Model Genetic Algorithm Attempting to Learn Optimal Reinforcement Learning Parameters:
    # Learning Rate & Epsilon (Included stochasiticity)

    # Create dictionary where key is graycode and value is the corresponding integer
    graycodedict = {}
    NUM_BITS = int(7)
    for i in range(0, 1 << NUM_BITS):
        gray = i ^ (i >> 1)
        graycodedict["{0:0{1}b}".format(gray, NUM_BITS)] = i

    print('Beginning Island Model GA.\n')

    #Initialize First Island Populations
    islandpopsize = 20
    chromolen = 14
    numislands = 4
    islandgenerations = 5
    migrations = 10
    successfulmigrations = 0
    bestfitnesslist = []
    EpisodesWindyWorld = 50
    # Initialize globally optimal values
    bestindividual, bestfitness, bestLR, bestEps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 100000, 0, 0


    print('Island Population Size: {}'.format(islandpopsize))
    print('Island Generation: {}'.format(islandgenerations))
    print('Number of Islands: {}'.format(numislands))
    print('Number of Migrations: {}\n'.format(migrations))

    # Randomly initialize population
    Pop = np.random.randint(2, size=(islandpopsize*numislands, chromolen))
    Fitness = np.zeros(islandpopsize*numislands)
    Island1Pop = Pop[0:islandpopsize]
    Island2Pop = Pop[islandpopsize:2*islandpopsize]
    Island3Pop = Pop[2*islandpopsize:3*islandpopsize]
    Island4Pop = Pop[3*islandpopsize:4*islandpopsize]

    # Determine fitness of randomly initialized population for statistical comparison
    for individual in range(Pop.shape[0]):
        Fitness[individual], LearningRate, Epsilon = CalculateFitness(Pop[individual], graycodedict)
        if Fitness[individual] < bestfitness:
            bestfitness = int(Fitness[individual])
    bestfitnesslist.append(bestfitness)

    # Multiprocessing Parameters

    # Introduce manager for a shared variable return_dict across threads
    # return_dict contains values typically returned by the GA function
    manager = mp.Manager()
    return_dict = manager.dict()

    # Iterate over number of migrations
    for i in range(migrations):

        # Assign 4 islands as a list of processes for multiprocessing
        Processes = [mp.Process(target=GeneticAlgorithm,
                                args=(Island1Pop, bestindividual, bestfitness, bestLR, bestEps,
                                      1, graycodedict, return_dict)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island2Pop, bestindividual, bestfitness, bestLR, bestEps,
                                      2, graycodedict, return_dict)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island3Pop, bestindividual, bestfitness, bestLR, bestEps,
                                      3, graycodedict, return_dict)),
                     mp.Process(target=GeneticAlgorithm,
                                args=(Island4Pop, bestindividual, bestfitness, bestLR, bestEps,
                                      4, graycodedict, return_dict))]

        # Start multiprocessing
        for process in Processes:
            process.start()

        # Sync when finished
        for process in Processes:
            process.join()

        # Collect return values from dictionary for each island
        ReturnIsland1 = return_dict[1]
        ReturnIsland2 = return_dict[2]
        ReturnIsland3 = return_dict[3]
        ReturnIsland4 = return_dict[4]

        # Create list of islands to determine optimal fitness value discovered
        IslandResults = [ReturnIsland1, ReturnIsland2, ReturnIsland3, ReturnIsland4]

        for island in range(len(IslandResults)):
            if IslandResults[island][4] < bestfitness:
                bestindividual = IslandResults[island][1]
                bestLR = IslandResults[island][2]
                bestEps = IslandResults[island][3]
                bestfitness = IslandResults[island][4]

        bestfitnesslist.append(bestfitness)

        # Create list of optimal individuals for migration
        BestIndividualList = [ReturnIsland1[1], ReturnIsland2[1], ReturnIsland3[1], ReturnIsland4[1]]

        # For each individual randomly select a partner from a different island to mate with
        islandlist = []
        for i in range(numislands):
            parent1 = BestIndividualList[i]
            RandNum = np.random.randint(0, 4)
            while RandNum == i: RandNum = np.random.randint(0,4)
            parent2 = BestIndividualList[RandNum]

            children = crossover([parent1, parent2])

            migratoryinds = np.vstack((children, parent1))
            migratoryinds = np.vstack((migratoryinds, parent2))

            MigratoryFitness = np.zeros(4)

            #Calculate fitness of migratory individuals
            for individual in range(migratoryinds.shape[0]):
                MigratoryFitness[individual], LearningRate, Epsilon = \
                    CalculateFitness(migratoryinds[individual], graycodedict)

            # Select the two best individuals
            islandlist.append(migratoryinds[np.argsort(MigratoryFitness)[0]])
            islandlist.append(migratoryinds[np.argsort(MigratoryFitness)[1]])

        # Randomly replace population member with optimal migratory individuals
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[0]
        Island1Pop[np.random.randint(0, islandpopsize)] = islandlist[1]
        Island2Pop[np.random.randint(0, islandpopsize)] = islandlist[2]
        Island2Pop[np.random.randint(0, islandpopsize)] = islandlist[3]
        Island3Pop[np.random.randint(0, islandpopsize)] = islandlist[4]
        Island3Pop[np.random.randint(0, islandpopsize)] = islandlist[5]
        Island4Pop[np.random.randint(0, islandpopsize)] = islandlist[6]
        Island4Pop[np.random.randint(0, islandpopsize)] = islandlist[7]

        successfulmigrations += 1

        print('\nMigration {} Complete'.format(successfulmigrations))
        print('Overall Least Number of Steps: ', bestfitness)
        print('Overall Optimal Individual: ', bestindividual)
        print('Overall Optimal Learning Rate: ', bestLR)
        print('Overall Optimal Epsilon: ', bestEps, '\n\n')

    print('List of Optimal Fitness: ', bestfitnesslist)

    plt.figure()
    plt.plot(bestfitnesslist)
    plt.title('Spatial GA Fitness Improvement After Migration Periods')
    plt.xlabel('Migrations')
    plt.ylabel('Least Number of Steps Taken Over {} Episodes of  Windy World'.format(EpisodesWindyWorld))
    plt.show()



