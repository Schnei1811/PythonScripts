

String1 = "hello"
String2 = "billion"

def collect_letters(String):
	Common_letters = {}
	for i in range(len(String)):
		if String[i] in Common_letters:
			Common_letters[String[i]] += 1
		else:
			Common_letters[String[i]] = 1
	return Common_letters

Common_letters1 = collect_letters(String1)
Common_letters2 = collect_letters(String2)

Anagram = []

Anagram = {}

for letter in Common_letters1:
    if letter in Common_letters2:
        if Common_letters1[letter] >= Common_letters2[letter]:
            Anagram[letter] = Common_letters1[letter]

print(len(String1) - sum(Anagram.values()))
print(len(String2) - sum(Anagram.values()))


