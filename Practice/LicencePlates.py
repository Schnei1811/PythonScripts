#Question: Imagine looking at Polish cars’ licence plates and trying to find a word from
# the dictionary that includes all the letters from the licence plate. The shorter the word,
# the better. The licence plates start with two or three letters, then they are followed by
# 5 characters, from which at most 2 are letters, the rest are digits.

#Your goal is to write code that will find the shortest words for 1xdif-100 licence plates. You
# are given a dictionary.

#e.g. for the licence plate "RT 123SO" the shortest word would be "SORT", for "RC 10014":
# "CAR".






def comparechardict(worddict, licencedict):
    for key in licencedict:
        if key not in worddict or worddict[key] < licencedict[key]: return False
    return True


def binarysearch(inputlist, licencedict, licence):
    first = 0
    last = len(inputlist) - 1
    shortestwordlen = 100
    shortestword = ''

    while first <= last:
        mid = (first+last) // 2
        worddict = buildworddict(inputlist[mid])
        print(licence, inputlist[mid])
        if comparechardict(worddict, licencedict) == True:
                if len(inputlist[mid]) < shortestword:
                    shortestwordlen = len(inputlist[mid])
                    shortestword = inputlist[mid]
                if shortestwordlen == len(licence):
                    return shortestword
        elif inputlist[mid] < licence: last = mid -1
        else: first = mid + 1
    return shortestword


def buildworddict(word):
    worddict = {}
    for char in word:
        if char in worddict:
            worddict[char] += 1
        else:
            worddict[char] = 1
    return worddict



def shortestword(englishlist, licencelist):
    replacevalues = ' 0123456789'
    finallist = []
    for licence in licencelist:
        for char in licence:
            if char in replacevalues:
                licence = licence.replace(char, '')
        licence = licence.lower()
        licencedict = buildworddict(licence)
        shortword = binarysearch(englishlist, licencedict, licence)
        finallist.append(shortword)
    print(finallist)










englishlist = ['apple', 'candy', 'car', 'freaking', 'only', 'sandy', 'sort', 'umbrella']
licencelist = ['RT 123SO', 'RC 10014']


shortestword(englishlist, licencelist)


