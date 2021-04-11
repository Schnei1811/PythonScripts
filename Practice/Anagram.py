String1 = "hello"
String2 = "billion"

def collect_letters(String):
	Common_letters = {}
	for char in String:
		if char in Common_letters:
			Common_letters[char] += 1
		else:
			Common_letters[char] = 1
	return Common_letters

Common_letters1 = collect_letters(String1)
Common_letters2 = collect_letters(String2)

Anagram = {}

for letter in Common_letters1:
    if letter in Common_letters2:
        if Common_letters1[letter] >= Common_letters2[letter]:
            Anagram[letter] = Common_letters1[letter]

print(len(String1) - sum(Anagram.values()))
print(len(String2) - sum(Anagram.values()))


