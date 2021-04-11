

string = 'abppplee'
dict = {0:'able',1:'ale',2:'apple',3:'bale',4:'kangaroo'}
teststring = ''
maxwordlength = 0
maxword = ''
k = 0




for i in range(len(dict[0])):
    for k in range(len(string)):
        if dict[0][i] == string[k]: break































for i in range(len(dict[0])):
    dictspace = 0
    while k < len(string):
        if dict[0][dictspace] == string[k]:
            k = 0
            dictspace += 1
        if dictspace == len(dict[0]):
            maxwordlength = len(dict[0])
            maxword = dict[0]
            print(dict[0])




