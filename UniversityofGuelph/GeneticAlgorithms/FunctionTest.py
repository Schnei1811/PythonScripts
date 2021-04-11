import numpy as np
import random

def symbolicmutation(chromo, nummutations):
    loci = []
    for mut in range(nummutations):
        mutationlocation = np.random.randint(0, chromolen)
        loci.append(mutationlocation)
    for loc in range(len(loci)):
        locrand = random.randint(0, 5)
        while locrand != loc:
            chromo[loci[loc]] = locrand
            locrand = random.randint(0, 5)
    colourchromo = [masterdict[chromo[0]], masterdict[chromo[1]], masterdict[chromo[2]], masterdict[chromo[3]]]
    print('Random Point 1: {}, Random Point 2: {}'.format(loci[0], loci[1]))
    print("Symbolic Fixed Mutation Colours ", colourchromo)
    return

def swapmutation(chromo):
    #It is possible for to swap with self
    randpos1, randpos2 = random.randint(0, chromolen - 1), random.randint(0, chromolen - 1)
    pos1, pos2 = chromo[randpos1], chromo[randpos2]
    chromo[randpos1] = pos2
    chromo[randpos2] = pos1
    colourchromo = [masterdict[chromo[0]], masterdict[chromo[1]], masterdict[chromo[2]], masterdict[chromo[3]]]
    print('Random Point 1: {}, Random Point 2: {}'.format(randpos1, randpos2))
    print("Swap Mutation Colours ", colourchromo)
    return

def inversionmutation(chromo):
    inversechromo, invlist, invlist1, invlist2 = [], [], [], []
    indlist = chromo
    randpos1, randpos2 = random.randint(1, chromolen), random.randint(1, chromolen)
    if randpos1 == randpos2:
        print('Inversion randomed same position')
    elif randpos1 < randpos2:
        invlist = indlist[randpos1: randpos2]
        invlist.reverse()
        inversechromo.extend(indlist[0:randpos1])
        inversechromo.extend(invlist)
        inversechromo.extend(indlist[randpos2:])
        inversechromo = np.asarray(inversechromo)
        chromo = inversechromo
        colourchromo = [masterdict[chromo[0]], masterdict[chromo[1]], masterdict[chromo[2]], masterdict[chromo[3]]]
        print('Random Point 1: {}, Random Point 2: {}'.format(randpos1, randpos2))
        print("Inversion Mutation Colours ", colourchromo)
    else:
        invlist1, invlist2 = indlist[0:randpos2], indlist[randpos1:]
        invlist2.extend(invlist1)
        invlist2.reverse()
        inversechromo = invlist2[0:randpos2]
        inversechromo.extend(indlist[randpos2:randpos1])
        inversechromo.extend(invlist2[randpos2:])
        inversechromo = np.asarray(inversechromo)
        chromo = inversechromo
        colourchromo = [masterdict[chromo[0]], masterdict[chromo[1]], masterdict[chromo[2]], masterdict[chromo[3]]]
        print('Random Point 1: {}, Random Point 2: {}'.format(randpos1, randpos2))
        print("Inversion Mutation Colours ", colourchromo)
    return

def randomizedmutation(chromo):
    child1, count = [0, 1, 4, 5], 0
    print("Chromo: {}, Child {}".format(chromo, child1))
    for i in range(len(chromo)):
        if chromo[i] == child1[i]: count += 1
    if count == 4: child1 = np.random.randint(6, size=chromolen)
    print("Chromo: {}, Child {}".format(chromo, list(child1)))
    return

masterdict = {0: 'Black', 1: 'White', 2: 'Green', 3:'Blue', 4:'Red', 5:'Yellow'}

chromolen = 4
chromo = [0, 1, 4, 5]
colourchromo = [masterdict[chromo[0]], masterdict[chromo[1]], masterdict[chromo[2]], masterdict[chromo[3]]]

print("Input Colours ", colourchromo)


symbolicmutation(chromo, 2)

chromo = [0, 1, 4, 5]
swapmutation(chromo)

chromo = [0, 1, 4, 5]
inversionmutation(chromo)

chromo = [0, 1, 4, 5]
inversionmutation(chromo)

chromo = [0, 1, 4, 5]
randomizedmutation(chromo)