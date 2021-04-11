import numpy as np
import matplotlib.pyplot as plt

OneMaxSolution = "11111111111111111111111111111111"

CHROMOSOMELENGTH = 16
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
EPSILON = 0

# Mutation
# Choose Fixes or Per Locus Binary Mutation
FIXED_BINARY_MUTATION = False
FIXED_NUM_MUTATIONS = 4
VARIATION_FIXED_NUM_MUTATION = False

PER_LOCUS_BINARY_MUTATION = True
PER_LOCUS_MUTATION_PROB = 0.2

# Crossover
# Choose K-Point or Uniform Crossover
PROB_CROSSOVER = 0.2
SET_NUM_CHILDREN = 20
K_POINT_CROSSOVER = True
K_NUM_CROSSOVER = 4

UNIFORM_CROSSOVER = False
PER_LOCUS_CROSSOVER_PROB = 0.2

# Selection
# Choose FPS or Tournament Selection
NUM_SELECTED = 50
FITNESS_PROPORTION_SELECTION = False
TOURNAMENT_SELECTION = True
TOURNAMENT_SIZE = 4
if FITNESS_PROPORTION_SELECTION and TOURNAMENT_SELECTION == False: NUM_SELECTED = 0

# Elitism
# Choose Elitism
ELITISM = True
NUM_ELITE = 5
if ELITISM == False: NUM_ELITE = 0

if SET_NUM_CHILDREN + NUM_SELECTED + NUM_ELITE > POPULATION_SIZE:
    print('Error. Number of Children + Selected + Elite cannot be greater than Population Size')
    quit()

POP = np.random.randint(2, size=(POPULATION_SIZE, CHROMOSOMELENGTH))
FITNESS = np.zeros((POPULATION_SIZE, 1))
MAX_FITNESS = 0
MAX_FITNESS_LIST = []

PRINT_OPTIMAL_PER_EPOCH = False


def Plot(MAX_FITNESS_LIST):
    y = MAX_FITNESS_LIST
    x = []
    for i in range(len(MAX_FITNESS_LIST)):
        x.append(i + 1)

    plt.plot(x, y, label='Fitness Over Time')
    plt.ylim([min(MAX_FITNESS_LIST) - 0.05, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Fitness')
    plt.title('One Max Genetic Algorithm Fitness')
    plt.legend()
    plt.show()
    return

NUM_TEST_ITER = 30
TOTAL_MAX_FITNESS_LIST = []
TOTAL_MAX_CHROMOSOME_LIST = []

#for test_iter in range(NUM_TEST_ITER):
#    MAX_FITNESS = 0
for epoch in range(NUM_GENERATIONS):
    # print('Epoch: {}'.format(epoch))

    # Mutation
    for i in range(POPULATION_SIZE):
        Loci = []
        if FIXED_BINARY_MUTATION == True:
            # Option to add variation to a fixed number of mutations
            if VARIATION_FIXED_NUM_MUTATION == True:
                NUM_MUTATIONS = np.random.randint(FIXED_NUM_MUTATIONS - 1, FIXED_NUM_MUTATIONS + 1)
            else:
                NUM_MUTATIONS = FIXED_NUM_MUTATIONS
            # Randomize Mutation Locations and store in mutation list
            for k in range(NUM_MUTATIONS):
                MUTATION_LOCATIONS = np.random.randint(0, CHROMOSOMELENGTH)
                Loci.append(MUTATION_LOCATIONS)

        elif PER_LOCUS_BINARY_MUTATION == True:
            # Create Random Array of Length Chromosome
            PER_LOCUS_BINARY_MUTATION_RANDOM = np.random.random((CHROMOSOMELENGTH, 1))
            # Create Mutation list of locus with random value < Per Locus Mutation Prob and add to Loci List
            for j in range(PER_LOCUS_BINARY_MUTATION_RANDOM.shape[0]):
                if PER_LOCUS_BINARY_MUTATION_RANDOM[j] < PER_LOCUS_MUTATION_PROB: Loci.append(j)

        # Swap binary values of Mutation List
        for k in range(len(Loci)):
            if POP[i][Loci[k]] == 0: POP[i][Loci[k]] = 1
            elif POP[i][Loci[k]] == 1: POP[i][Loci[k]] = 0

    # Crossover
    NUM_CHILDREN = 0
    if np.random.random() < PROB_CROSSOVER:
        # Set Number of offspring
        NUM_CHILDREN = SET_NUM_CHILDREN
        CHILD_POPULATION = np.zeros((NUM_CHILDREN, CHROMOSOMELENGTH))
        for i in range(int(NUM_CHILDREN / 2)):
            # Randomly choose two parents
            PARENT1 = POP[np.random.randint(0, POPULATION_SIZE)]
            PARENT2 = POP[np.random.randint(0, POPULATION_SIZE)]
            LOCI = []

            # Randomize K Locations
            if K_POINT_CROSSOVER == True:
                for j in range(K_NUM_CROSSOVER): LOCI.append(np.random.randint(0, CHROMOSOMELENGTH))
                LOCI.sort()

            # Create Random Array and add loci with value < Per Locus Cross Prob to Loci List
            if UNIFORM_CROSSOVER == True:
                PER_LOCUS_CROSSOVER_RANDOM = np.random.random((CHROMOSOMELENGTH, 1))
                for j in range(PER_LOCUS_CROSSOVER_RANDOM.shape[0]):
                    if PER_LOCUS_CROSSOVER_RANDOM[j] < PER_LOCUS_CROSSOVER_PROB: LOCI.append(j)

            if len(LOCI) == 0:
                pass
            elif len(LOCI) == 1:
                CHILD1 = np.append(PARENT1[0:LOCI[0]], PARENT2[LOCI[0]:])
                CHILD2 = np.append(PARENT2[0:LOCI[0]], PARENT1[LOCI[0]:])
                # Store Child Population in array
                CHILD_POPULATION[i] = CHILD1
                CHILD_POPULATION[i + int(NUM_CHILDREN / 2)] = CHILD2

            else:
                for j in range(len(LOCI)):
                    if j == 0:
                        CHILD1 = PARENT1[0:LOCI[j]]
                        CHILD2 = PARENT2[0:LOCI[j]]
                    elif j % 2 == 0 and j == len(LOCI) - 1:
                        CHILD1 = np.append(CHILD1, PARENT1[LOCI[j - 1]:])
                        CHILD2 = np.append(CHILD2, PARENT2[LOCI[j - 1]:])
                    elif j % 2 == 1 and j == len(LOCI) - 1:
                        CHILD1 = np.append(CHILD1, PARENT2[LOCI[j - 1]:])
                        CHILD2 = np.append(CHILD2, PARENT1[LOCI[j - 1]:])
                    elif j % 2 == 0:
                        CHILD1 = np.append(CHILD1, PARENT1[LOCI[j - 1]:LOCI[j]])
                        CHILD2 = np.append(CHILD2, PARENT2[LOCI[j - 1]:LOCI[j]])
                    elif j % 2 == 1:
                        CHILD1 = np.append(CHILD1, PARENT2[LOCI[j - 1]:LOCI[j]])
                        CHILD2 = np.append(CHILD2, PARENT1[LOCI[j - 1]:LOCI[j]])
                        # Store Child Population in array
                CHILD_POPULATION[i] = CHILD1
                CHILD_POPULATION[i + int(NUM_CHILDREN / 2)] = CHILD2

    # Determine Fitness & Build Elite Population
    if ELITISM == True:
        ELITISM_FITNESS = np.zeros((1, NUM_ELITE))[0]
        ELITISM_INDIVIDUAL = np.zeros((1, NUM_ELITE))[0]
        ELITISM_POPULATION = np.zeros((NUM_ELITE, CHROMOSOMELENGTH))

    PER_EPOCH_MAX_FITNESS = 0

    for i in range(POPULATION_SIZE):
        # Calculate Fitness
        FITNESS[i] = np.divide(sum(POP[i]), CHROMOSOMELENGTH)
        # Keep Track of Max Fitness and Solution
        if float(FITNESS[i]) > PER_EPOCH_MAX_FITNESS:
            PER_EPOCH_MAX_FITNESS = float(FITNESS[i])
            PER_EPOCH_OPTIMAL_CHROMOSOME = POP[i]
        if float(FITNESS[i]) > MAX_FITNESS:
            MAX_FITNESS = float(FITNESS[i])
            OPTIMAL_SOLUTION = POP[i]
            #print('New Optimal: ', epoch, float(FITNESS[i]), OPTIMAL_SOLUTION)
        # Print and quit if solution found
        if FITNESS[i] == 1 - EPSILON:
            #print('SOLUTION FOUND! Epoch {}. Optimal Solution {}. '
            #      'Fitness {}'.format(epoch, OPTIMAL_SOLUTION, float(FITNESS[i])))
            MAX_FITNESS_LIST.append(PER_EPOCH_MAX_FITNESS)
        if ELITISM == True:
            # Add first 10 population to Elite array
            if i < NUM_ELITE:
                ELITISM_FITNESS[i] = FITNESS[i]
                ELITISM_INDIVIDUAL[i] = i
            else:
                # If new fitness larger than smallest fitness in Elite array, replace.
                if FITNESS[i] > np.min(ELITISM_FITNESS):
                    ELITISM_FITNESS[np.argmin(ELITISM_FITNESS)] = FITNESS[i]
                    ELITISM_INDIVIDUAL[np.argmin(ELITISM_FITNESS)] = i
            if i == POPULATION_SIZE - 1:
                # Store Elite Population in array
                for k in range(NUM_ELITE): ELITISM_POPULATION[k] = POP[int(ELITISM_INDIVIDUAL[k])]

    # Build list of max fitnesses for plot
    MAX_FITNESS_LIST.append(PER_EPOCH_MAX_FITNESS)
    if PRINT_OPTIMAL_PER_EPOCH == True:
        print('Chromosome: {}, Fitness: {}, Epoch: {}'.format(PER_EPOCH_OPTIMAL_CHROMOSOME,
                                                              PER_EPOCH_MAX_FITNESS, epoch))

    # Population Selection
    SELECTED_POPULATION = np.zeros((NUM_SELECTED, CHROMOSOMELENGTH))

    if FITNESS_PROPORTION_SELECTION == True:
        # Create cumulative array of fitness values. Individuals with larger fitness will accumulate larger values.
        FITNESS_SUM = np.cumsum(FITNESS / sum(FITNESS))
        for k in range(NUM_SELECTED):
            # For the pre-defined number selected, return the closest without going over of a random value
            SELECTED_INDIVIDUAL = FITNESS_SUM[(FITNESS_SUM <= np.random.random())]
            # Store Selected Population in array
            SELECTED_POPULATION[k] = POP[len(SELECTED_INDIVIDUAL)]

    if TOURNAMENT_SELECTION == True:
        for k in range(NUM_SELECTED):
            TOURNAMENT_LIST, TOURNAMENT_FITNESS = [], []
            # Randomize Tournament Entries
            for j in range(TOURNAMENT_SIZE): TOURNAMENT_LIST.append(np.random.randint(0, POPULATION_SIZE))
            # Determine fitness of each entry
            for j in range(TOURNAMENT_SIZE): TOURNAMENT_FITNESS.append(np.divide(sum(POP[TOURNAMENT_LIST[j]]), 32))
            # Store Selected Population in Array
            SELECTED_POPULATION[k] = POP[TOURNAMENT_FITNESS.index(max(TOURNAMENT_FITNESS))]

    # Initialize new population considering the total population size and stored population
    NEW_INIT_POP = np.random.randint(2, size=(POPULATION_SIZE - NUM_SELECTED - NUM_ELITE - NUM_CHILDREN,
                                              CHROMOSOMELENGTH))

    # Join to create new population
    if ELITISM == True:
        NEW_POP = np.vstack((SELECTED_POPULATION, ELITISM_POPULATION))
    else:
        NEW_POP = SELECTED_POPULATION
    if NUM_CHILDREN > 0: NEW_POP = np.vstack((NEW_POP, CHILD_POPULATION))
    NEW_POP = np.vstack((NEW_POP, NEW_INIT_POP))

    # Shuffle for added randomness
    np.random.shuffle(NEW_POP)
    POP = NEW_POP

TOTAL_MAX_FITNESS_LIST.append(MAX_FITNESS)
TOTAL_MAX_CHROMOSOME_LIST.append(OPTIMAL_SOLUTION)
print("Average Fitness: ", sum(FITNESS) / POPULATION_SIZE)
print("Max Fitness: ", MAX_FITNESS)
print("Optimal Solution: ", OPTIMAL_SOLUTION)
#Plot(MAX_FITNESS_LIST)


#print(TOTAL_MAX_FITNESS_LIST)
#print("Average Fitness: ", np.mean(TOTAL_MAX_FITNESS_LIST))
#for i in range(len(TOTAL_MAX_CHROMOSOME_LIST)):
#    print("{} Fitness: {}".format(TOTAL_MAX_CHROMOSOME_LIST[i], TOTAL_MAX_FITNESS_LIST[i]))























































