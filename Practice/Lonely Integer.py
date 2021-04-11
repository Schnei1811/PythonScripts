

Array = [2, 7, 1, 2, 5, 7, 1]


Array.sort()

def lonely_int_fnc(Array):

    for i in range(len(Array)):
        print(Array, i)
        if Array[i] != Array[i+1]:
            return Array[i]
        i += 2

print(lonely_int_fnc(Array))

