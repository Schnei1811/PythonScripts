

string1 = "hello"
string2 = "baggageo"

set1 = set(string1)
set2 = set(string2)


print(set2-set1)        # letters in string 2 but not string 1
print(set2&set1)        # letters in string 2 and string 1
print(set2|set1)        # letters in string 2 and or string 1
print(set2^set1)        # letters not in string 2 or string 1

print(25**0.5)