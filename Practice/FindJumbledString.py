

def check(sub, full):
    full_list = list(full)
    for char in sub:
        if char in full_list:
            full_list.remove(char)
        else:
            return False
    return True


string1 = 'rtsop'
string2 = 'sort'

print(check(string1, string2))

string = 'DFA'
string = string.lower()
print(string)



def checkwordchars(jumble, string):
    word_list = list(string)
    for char in jumble:
        if char in word_list:
            word_list.remove(char)
        else:
            return False
    return True

print(checkwordchars(string1, string2))












def binarysearch(alist, val):
    first = 0
    last = len(alist) - 1

    while first <= last:
        mid = (first + last) // 2
        if alist[mid] == val: return True
        elif alist[mid] > val: last = mid - 1
        else: first = mid + 1
    return False




def containjumblestring(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True

print(containjumblestring('srd', 'serdene'))








def containjumble(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True

print(containjumble('afd','afrdic'))











def containedjumble(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True























def containslist(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True













def checkmixedstring(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True

print(checkmixedstring('gd', 'goodness'))






















def checkmixedstring(jumble, string):
    string = list(string)
    for char in jumble:
        if char in string:
            string.remove(char)
        else:
            return False
    return True






string = 'string'
for char in string:
    print(char)









print(checkmixedstring('gd', 'goodness'))




















