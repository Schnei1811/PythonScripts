lines = open('ABC/Zelda3Dungeon.abc').readlines()
s = ''
for line in lines[9:]:
    if not line.startswith('%'):
        s += line
s = s.replace('M: ','')
s = s.replace('Q: ',' ')
s = s.replace('K: ',' ')
s = s.replace('C maj','Cmaj ')
s = s.replace('|','| ')
s = s.replace('\n','')