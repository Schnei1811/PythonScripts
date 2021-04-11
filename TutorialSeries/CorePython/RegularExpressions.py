import re



text_to_search = '''

abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ

Ha HaHa

MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] / | ( )

coreyms.com

321-555-4321
123.555.1234

Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''


sentence = 'Start a setence and then bring it to an end'

print('\tTab')

print(r'\tTab')

pattern = re.compile(r'abc')
#pattern = re.compile(r'.')         #without backslash uses regex .
#pattern = re.compile(r'\.')         #to look for period
pattern = re.compile(r'coreyms\.com')


matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)

print(text_to_search[1:4])










































