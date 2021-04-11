import random as random
import numpy as np


def calc_pins(guess):
    testanswer = list(masteranswer)
    blackpins, whitepins = 0, 0
    for i in range(len(guess)):
        if guess[i] == testanswer[i]:
            testanswer[i] = 7
            blackpins += 1
    for i in range(len(guess)):
        if guess[i] in testanswer:
            testanswer[testanswer.index(guess[i])] = 7
            whitepins += 1
    return blackpins, whitepins

def calc_fitness(guess, turn):
    blackpins, whitepins = calc_pins(guess)
    if turn == 0: fitness = alpha * blackpins + whitepins + beta * P * (1 - 1)
    else:
        Xsum, Ysum = 0, 0
        for i in range(turn):
            Xsum += blackpins - pinlist[i][0]
            Ysum += whitepins - pinlist[i][1]
        fitness = alpha * Xsum + Ysum + beta * P * (turn - 1)
    return blackpins, whitepins, fitness

def fixedsymbolicmutation():
    loci = []
    for mut in range(nummutations):
        mutationlocation = np.random.randint(0, chromolen)
        loci.append(mutationlocation)
    return loci

def perlocisymbolicmutation():
    loci = []
    perlocisymbolicmutrand = np.random.random((chromolen, 1))
    for prob in range(perlocisymbolicmutrand.shape[0]):
        if perlocisymbolicmutrand[prob] <= perlocimutprob:
            loci.append(prob)
    return loci


masterdict = {0: 'Black', 1: 'White', 2: 'Green', 3:'Blue', 4:'Red', 5:'Yellow'}
alpha, beta, P = 1, 2, 4
numturns, numtrials, correctanswers = 12, 100, 0
turnlist = []

masteranswer = (random.randint(0, 5), random.randint(0, 5), random.randint(0, 5), random.randint(0, 5))
guesslist = [[1, 1, 2, 3]]
pinlist = []

popsize = 100
generations = 100
chromolen = 4
allelenum = 6


#Randomly initialize population
pop = np.random.randint(6, size=(popsize, chromolen))
fitness = np.zeros((popsize, 1))
maxfitness, minfitness = 0, 1000
maxfitnesslist, minfitnesslist = [], []


#Mutation
fixedsymbolicmut = False
nummutations = 1

perlocisymbolicmut = False
perlocimutprob = 0.3

swapmut = False
swapmutprob = 0.3

inversionmut = True
inversionmutprob = 0.2

randomizedmut = False

#Crossover
probcrossover = 0.2
crossnumchildren = 20

kpointcrossover = True
knumcrossover = 2

uniformcrossover = False
perlocicrossprob = 0.1

#Elitism
elitism = True
numelite = 2

#Selection
numselected = 40

fps = False

tournselect = True
tournsize = 2

print('Initiating 1xdif-100 Trials of GA Solving Mastermind')

for trial in range(numtrials):
    correctanswer = False
    print('Trial Number {}'.format(trial))
    for turn in range(numturns):
        maxfitness = 0
        for epoch in range(generations):
            for ind in range(popsize):
                if fixedsymbolicmut == True:
                    loci = fixedsymbolicmutation()
                    for loc in range(len(loci)):
                        locrand = random.randint(0, 5)
                        while locrand != loc:
                            pop[ind][loci[loc]] = locrand
                            locrand = random.randint(0, 5)

                if perlocisymbolicmut == True:
                    loci = perlocisymbolicmutation()
                    for loc in range(len(loci)):
                        locrand = random.randint(0, 5)
                        while locrand != loc:
                            pop[ind][loci[loc]] = locrand
                            locrand = random.randint(0, 5)

                if swapmut == True and np.random.random() < swapmutprob:
                    randpos1, randpos2 = random.randint(0, chromolen-1), random.randint(0, chromolen-1)
                    pos1, pos2 = pop[ind][randpos1], pop[ind][randpos2]
                    pop[ind][randpos1] = pos2
                    pop[ind][randpos2] = pos1

                if inversionmut == True and np.random.random() < inversionmutprob:
                    inversechromo, invlist, invlist1, invlist2 = [], [], [], []
                    indlist = pop[ind].tolist()
                    randpos1, randpos2 = random.randint(1, chromolen), random.randint(1, chromolen)
                    if randpos1 == randpos2: pass
                    elif randpos1 < randpos2:
                        invlist = indlist[randpos1: randpos2]
                        invlist.reverse()
                        inversechromo.extend(indlist[0:randpos1])
                        inversechromo.extend(invlist)
                        inversechromo.extend(indlist[randpos2:])
                        inversechromo = np.asarray(inversechromo)
                        pop[ind] = inversechromo
                    else:
                        invlist1, invlist2 = indlist[0:randpos2], indlist[randpos1:]
                        invlist2.extend(invlist1)
                        invlist2.reverse()
                        inversechromo = invlist2[0:randpos2]
                        inversechromo.extend(indlist[randpos2:randpos1])
                        inversechromo.extend(invlist2[randpos2:])
                        inversechromo = np.asarray(inversechromo)
                        pop[ind] = inversechromo


            numchildren = 0
            if np.random.random() < probcrossover:
                numchildren = crossnumchildren
                childpop = np.zeros((numchildren, chromolen))
                for child in range(numchildren // 2):
                    parent1 = pop[np.random.randint(0, popsize)]
                    parent2 = pop[np.random.randint(0, popsize)]
                    loci = []

                    if kpointcrossover == True:
                        for j in range(knumcrossover): loci.append(np.random.randint(0, chromolen))
                        loci.sort()

                    if uniformcrossover == True:
                        perlocicrossrand = np.random.random((chromolen, 1))
                        for j in range(perlocicrossrand.shape[0]):
                            if perlocicrossrand[j] < perlocicrossprob: loci.append(j)

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

                        if randomizedmut == True:
                            for i in range(len(childpop)):
                                if np.array_equal(childpop[i], child1): child1 = np.random.randint(6, size=chromolen)
                                if np.array_equal(childpop[i], child2): child2 = np.random.randint(6, size=chromolen)

                        childpop[child] = child1
                        childpop[child + int(numchildren // 2)] = child2

            if elitism == True:
                elitefitness = np.zeros((1, numelite))[0]
                eliteindividual = np.zeros((1, numelite))[0]
                elitepop = np.zeros((numelite, chromolen))

            for ind in range(popsize):
                blackpins, whitepins, fitness[ind] = calc_fitness(pop[ind], turn)
                if int(fitness[ind]) > maxfitness:
                    maxfitness = int(fitness[ind])
                    bestguess = pop[ind]
                    maxblackpins = blackpins
                    maxwhitepins = whitepins
                if elitism == True:
                    if ind < numelite:
                        elitefitness[ind] = fitness[ind]
                        eliteindividual[ind] = ind
                    else:
                        if fitness[ind] > np.min(elitefitness):
                            elitefitness[np.argmin(elitefitness)] = fitness[ind]
                            eliteindividual[np.argmin(elitefitness)] = fitness[ind]
                        if ind == popsize - 1:
                            for k in range(numelite):
                                elitepop[k] = pop[int(eliteindividual[k])]

            selectedpop = np.zeros((numselected, chromolen))

            if fps == True:
                fpsfitness = np.cumsum(fitness / sum(fitness))
                for k in range(numselected):
                    fpsind = fpsfitness[(fpsfitness <= np.random.random())]
                    # Store Selected Population in array
                    selectedpop[k] = pop[len(fpsind)]

            if tournselect == True:
                for k in range(numselected):
                    tournlist, tournfitness = [], []
                    for j in range(tournsize):
                        tournlist.append(np.random.randint(0, popsize))
                    for j in range(tournsize):
                        tournfitness.append(np.divide(sum(pop[tournlist[j]]), 32))
                    selectedpop[k] = pop[tournfitness.index(max(tournfitness))]

            newinitpop = np.random.randint(6, size=(popsize - numselected - numelite - numchildren, chromolen))
            if elitism == True: newpop = np.vstack((selectedpop, elitepop))
            else: newpop = selectedpop
            if numchildren > 0: newpop = np.vstack((newpop, childpop))
            newpop = np.vstack((newpop, newinitpop))
            np.random.shuffle(newpop)
            pop = newpop

        #Make Guess
        guess = bestguess
        pinlist.append([maxblackpins, maxwhitepins])

        # print(turn)
        # print('Answer ', masteranswer)
        # print('Guess ', bestguess)

        counter = 0
        for i in range(len(guess)):
            if guess[i] == masteranswer[i]: counter += 1
            if counter == 4 and correctanswer == False:
                correctanswers += 1
                turnlist.append(turn)
                print('Answer Found! Turn: {} Total Number Correct Answers: {}'.format(turn, correctanswers))
                correctanswer = True







