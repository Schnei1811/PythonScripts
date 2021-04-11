import numpy as np
import ast
from sklearn.cross_validation import train_test_split
from sklearn import tree
np.set_printoptions(threshold=np.nan)
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import numpy as np

# Convert Midi to ABC

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    global snnepoch
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for i in range(m):
        try:
            first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        except ValueError:
            print("Check if you have classifications that are 0 or if num_labels is correct")
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]
        d3t = ht - yt
        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    print("Iteration:",snnepoch, " Cost:",J)
    snnepoch = snnepoch +1
    return J, grad

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J

def CreateDictionary():
    outputscale, inputscale = {},{}
    i = 0
    n = 3
    while i < abcdata.shape[0]:
        while n < LengthMusic+3:
            note = abcdata[i,n]
            try:
                foo = inputscale[note]
            except KeyError:
                outputscale[len(outputscale)] = str(note)
                inputscale = {v: k for k, v in outputscale.items()}
            n = n + 1
        i = i + 1
    print('Output Scale: ',outputscale)
    print('Length Output Scale: ',len(outputscale))
    return outputscale, inputscale

def TrainRandomForest(newsong_data, NotesCreated, minusindex,data, startingnote, newsong):
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:cols]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_y = np.ravel(train_y)
    rfclf.fit(train_x,train_y)
    newsong_data = newsong_data.reshape(1,-1)
    intnewnote = int(rfclf.predict(newsong_data))
    newnote = outputscale[int(rfclf.predict(newsong_data))]
    char = ''
    if len(newsong) > 5:
        if any(i  in newsong[len(newsong)- minusindex] for i in addingties):
            print(newsong[len(newsong)- minusindex])
            if any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 6:
                minusindex = 7
                print(newsong[len(newsong)- minusindex])
                print(len(newsong))
                print('7')
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 5:
                minusindex = 6
                print('6')
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 4:
                minusindex = 5
                print('5')
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 3:
                minusindex = 4
                print('4')
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 2:
                minusindex = 3
                print('3')
            elif any(i in (newsong[len(newsong) - minusindex]) for i in addingextranoteslist) and minusindex == 1:
                minusindex = 2
                print('2')
            else:
                minusposition = (newsong[len(newsong) - minusindex].index('-'))
                newnotevar1 = newsong[len(newsong) - minusindex][0:minusposition]
                newnotevar2 = newsong[len(newsong) - minusindex][(minusposition+1):]
                newnote = newnotevar1 + newnotevar2
                if any(i  in newnote for i in addingties) > 0:
                    minusposition = newnote.index('-')
                    newnotevar1 = newnote[0:minusposition]
                    newnotevar2 = newnote[(minusposition+1):]
                    newnote = newnotevar1 + newnotevar2
                    print(newnote)
                    if any(i  in newnote for i in addingties):
                        minusposition = newnote.index('-')
                        newnotevar1 = newnote[0:minusposition]
                        newnotevar2 = newnote[(minusposition+1):]
                        newnote = newnotevar1 + newnotevar2
                        print(newnote)
                        if any(i in newnote for i in addingties):
                            minusposition = newnote.index('-')
                            newnotevar1 = newnote[0:minusposition]
                            newnotevar2 = newnote[(minusposition+1):]
                            newnote = newnotevar1 + newnotevar2
                            print(newnote)

    if len(newsong) == NotesCreated-1 and any(i in newnote for i in addingties):
        minusposition = newnote.index('-')
        newnotevar1 = newnote[0:minusposition]
        newnotevar2 = newnote[(minusposition + 1):]
        newnote = newnotevar1 + newnotevar2
        if any(i in newnote for i in addingties):
            minusposition = newnote.index('-')
            newnotevar1 = newnote[0:minusposition]
            newnotevar2 = newnote[(minusposition + 1):]
            newnote = newnotevar1 + newnotevar2
            print(newnote)

    if any(i in newnote for i in addingextranoteslist) and NotesCreated == startingnote + 1:
        NotesCreated = NotesCreated + 1
    newsong[len(newsong)] = newnote
    newsong_data = np.append(newsong_data,np.zeros((1,UniqueCharacter)))
    newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
    #print(newsong)
    #print(round(startingnote / LengthMusic,2),"%")
    return newsong_data, NotesCreated, minusindex

def TrainSimpleNN(newsong_data, NotesCreated, data, startingnote):
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:cols]

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    input_size = train_x.shape[1]
    hidden_size = 500
    regularization = 1
    maxiter = 100
    encoder = OneHotEncoder(sparse=False)
    num_labels = len(np.unique(y))

    train_y_onehot = encoder.fit_transform(train_y)
    y_onehot = np.matrix(train_y_onehot)
    X = np.matrix(train_x)

    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    print("Training...")
    fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, regularization),
                method='TNC', jac=True, options={'maxiter': maxiter})
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    X = np.matrix(newsong_data)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    intnewnote = int(np.array(np.argmax(h, axis=1) + 1))
    intnewnote = np.unique(y)[intnewnote-1]
    newnote = outputscale[intnewnote]
    if any(i in newnote for i in addingextranoteslist) and NotesCreated == startingnote+1:
        NotesCreated = NotesCreated + 1
    newsong[len(newsong)] = newnote
    newsong_data = np.append(newsong_data, np.zeros((1, UniqueCharacter)))
    newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
    print(newsong)
    global snnepoch
    snnepoch = 0
    return newsong_data, NotesCreated

def Preprocess():
    lines = open(dataname[0]).readlines()
    s = ''
    for line in lines[9:]:
        if not line.startswith('% B'):
            s += line
    s = s.replace('M: ', '')
    s = s.replace('Q: ', ' ')
    s = s.replace('K: ', ' ')
    s = s.replace('C maj', 'Cmaj ')
    s = s.replace('|', '| ')
    s = s.replace('\n', '')
    s = s.replace('\t', ' ')
    s = s.split(' ')
    abcdata = np.asarray(s)

    i = 1
    while i < len(dataname):
        lines = open(dataname[i]).readlines()
        s = ''
        for line in lines[9:]:
            if not line.startswith('%'):
                s += line
        s = s.replace('M: ', '')
        s = s.replace('Q: ', ' ')
        s = s.replace('K: ', ' ')
        s = s.replace('C maj', 'Cmaj ')
        s = s.replace('|', '| ')
        s = s.replace('\n', '')
        s = s.replace('\t', ' ')
        s = s.split(' ')
        newabcdata = np.asarray(s)
        if abcdata.shape[0] > newabcdata.shape[0]:
            while abcdata.shape[0] > newabcdata.shape[0]:
                newabcdata = np.append([newabcdata], [newabcdata[numdescriptors:]])  # Tempo, Timing, Key
                newabcdata = newabcdata[0:abcdata.shape[0]]
        else:
            while abcdata.shape[0] < newabcdata.shape[0]:
                abcdata = np.append([abcdata], [abcdata[numdescriptors:]])  # Tempo, Timing, Key
                abcdata = abcdata[0:newabcdata.shape[0]]

        abcdata = np.append([abcdata], [newabcdata], axis=0)
        i = i + 1
    print('Num Notes: ', abcdata.shape[1])
    # flatabcdata = abcdata.flatten()
    # i = 0      #
    # while i < factorrowexpansion:
    #     flatabcdata = np.insert(flatabcdata,len(flatabcdata),flatabcdata)
    #     i = i + 1
    return abcdata

def BuildOneHot():
    input_data = np.zeros(shape=(abcdata.shape[0], num_features + lendescriptors), dtype=np.int)
    i = 0
    n = 0
    while i < abcdata.shape[0]:
        input_data[i,0] = Tempo[i]
        input_data[i,inputtimemetric[Timing[i]]] = 1
        input_data[i,inputkey[Key[i]]+len(outputtimemetric)] = 1
        k = 0
        while n < LengthMusic:
            note = abcdata[i,(n+numdescriptors)]
            input_data[i,k+inputscale[note]+lendescriptors]=1
            n = n + 1
            k = k + UniqueCharacter
        n=0
        i = i + 1
    return input_data

def CreateTrack(NotesCreated):
    minusindex = 1
    songtempo, songtiming, songkey = abcdata[0, 1], abcdata[0, 0], abcdata[0, 2]
    note1, note2, note3 = outputscale[0], outputscale[1], outputscale[2]
    outputstring = note1 + ' ' + note2 + ' ' + note3
    newsong = {0: note1, 1: note2, 2: note3}
    songtiming = inputtimemetric[songtiming]
    songkey = inputkey[songkey]
    note1 = inputscale[note1]
    note2 = inputscale[note2]
    note3 = inputscale[note3]
    notes = {0: note1, 1: note2, 2: note3}
    startingnote = 3
    newsong_data = np.zeros((1, lendescriptors + UniqueCharacter * startingnote))
    newsong_data[0, 0] = songtempo
    newsong_data[0, songtiming] = 1
    newsong_data[0, len(outputtimemetric) + songkey] = 1
    newsong_data[0, lendescriptors + UniqueCharacter * 0 + notes[0]] = 1
    newsong_data[0, lendescriptors + UniqueCharacter * 1 + notes[1]] = 1
    newsong_data[0, lendescriptors + UniqueCharacter * 2 + notes[2]] = 1
    while startingnote < NotesCreated:
        y = np.zeros((abcdata.shape[0], 1))
        X = input_data[:, 0:lendescriptors + UniqueCharacter * startingnote]
        ynoteonehot = input_data[:, lendescriptors + UniqueCharacter * startingnote:lendescriptors + (startingnote * UniqueCharacter) + UniqueCharacter]
        n = 0
        i = 0
        while i < abcdata.shape[0]:
            while n < UniqueCharacter:
                if ynoteonehot[i, n] == 1:
                    newvar = n / UniqueCharacter
                    note = inputscale[outputscale[round((newvar % 1) * UniqueCharacter)]]
                    y[i] = int(note)
                n = n + 1
            n = 0
            i = i + 1
        data = np.append(X, y, axis=1)
        newsong_data, NotesCreated, minusindex = TrainRandomForest(newsong_data, NotesCreated, minusindex, data, startingnote, newsong)
        # newsong_data, NotesCreated = TrainSimpleNN(newsong_data, NotesCreated)
        startingnote = startingnote + 1
    return newsong_data, newsong, outputstring

def OutputFile(outputstring):
    OutTempo = newsong_data[0]

    n = 1
    while n < len(outputtimemetric) + 1:
        if newsong_data[n] == 1:
            OutTiming = outputtimemetric[n]
        n = n + 1

    n = len(outputtimemetric)
    while n < len(outputtimemetric) + len(inputkey) + 1:
        if newsong_data[n] == 1:
            OutKey = outputkey[n - len(outputtimemetric)]
        n = n + 1

    print(newsong)
    i = 3
    while i < len(newsong):
        outputstring = outputstring + ' ' + newsong[i]
        i = i + 1

    text_file = open("Test.txt", 'w')
    text_file.write(
        "M: " + str(OutTiming) + "\nQ: " + str(int(OutTempo)) + "\nK: " + str(OutKey) + "\n\n" + outputstring)
    text_file.close()


rfclf = tree.DecisionTreeClassifier()
snnepoch = 0

addingextranoteslist = ('|','+mp+','+mf+','+f+','+ff+','+fff+','+ffff+','+p+','+pp+','+ppp+','+pppp+')
addingties = ('-')
outputinstrument = {1:'Drums',2:'Theorbo',3:'LuteofAges',4:'Clarinet',5:'Flute'}
outputtimemetric = {1:'4/4',2:'3/4',3:'1/4',4:'2/4',5:'5/8',6:'6/8'}
outputkey = {1:'Cmaj'}
outputsetting = {1:'Field',2:'Town',3:'Desert',4:'Ice',5:'Battle',6:'Boss'}
outputmood = {1:'Happy',2:'Sad'}

inputinstrument = {v: k for k, v in outputinstrument.items()}
inputtimemetric = {v: k for k, v in outputtimemetric.items()}
inputkey = {v: k for k, v in outputkey.items()}
inputsetting = {v: k for k, v in outputsetting.items()}
inputmood = {v: k for k, v in outputmood.items()}

lendescriptors = 1 + len(outputtimemetric) + len(outputkey)
numdescriptors = 3

#dataname = {0:'ABCFav/Zelda3DarkWorld.abc',1:'ABCFav/Zelda3DarkWorld.abc'}
#dataname = {0:'ABCFav/SuperMarioRPGForestMaze.abc',1:'ABCFav/SuperMarioRPGForestMaze.abc'}
#dataname = {0:'ABCFav/Zelda3Dungeon.abc',1:'ABCFav/Zelda3Dungeon.abc'}
#dataname = {0:'ABCFav/CronoTriggerEpoch.abc',1:'ABCFav/CronoTriggerEpoch.abc'}
#dataname = {0:'ABCFav/CronoTriggerForest.abc',1:'ABCFav/CronoTriggerForest.abc'}
dataname = {0:'ABCFav/ActraiserBoss.abc',1:'ABCFav/ActraiserBoss.abc'}
#dataname = {0:'ABCFav/EarthboundDeepMagicent.abc',1:'ABCFav/EarthboundDeepMagicent.abc'}

#dataname = {0:'ABCFav/DKC1AquaticAmbience.abc',1:'ABCFav/EarthBoundDeepMagicent.abc'}
#dataname = {0:'ABCFav/EarthBoundSoundStone.abc',1:'ABCFav/EarthBoundDeepMagicent.abc'}
#dataname = {0:'ABCFav/EarthBoundSailing.abc',1:'ABCFav/CronoTriggerEpoch.abc'}
#dataname = {0:'ABC/EarthboundThreed.abc',1:'ABC/EarthboundTwoson.abc'}
#dataname = {0:'ABCFav/SuperMarioRPGForestMaze.abc',1:'ABC/FinalFantasy6HauntedForest.abc'}

factorrowexpansion = 5                                                      #10-12 good
NotesCreated = 100

abcdata = Preprocess()

Timing = abcdata[:,0]
Tempo = abcdata[:,1]
Key = abcdata[:,2]

LengthMusic = abcdata.shape[1]-numdescriptors                                            #3 = Tempo, TimeMetric, Key

outputscale, inputscale = CreateDictionary()

UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic

input_data = BuildOneHot()

newsong_data,newsong,outputstring = CreateTrack(NotesCreated)

OutputFile(outputstring)