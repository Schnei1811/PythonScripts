


string = "aaabbdeeeaaadvc"

compressedstring = ""

i = 0
while i < len(string):
    counter = 1
    while True:
        currentpos = i
        if string[currentpos] == string[i + 1]:
            counter += 1
            i += 1
        else:
            compressedstring += string[currentpos]
            compressedstring += str(counter)
            i += 1
            break

print(compressedstring)







