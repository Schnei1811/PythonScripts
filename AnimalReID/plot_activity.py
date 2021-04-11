import matplotlib.pyplot as plt
import numpy as np

def t(a, b):
    return a*60 + b

octo_dct = {
    'oct1': [[1, 4, 3, 4, 3, 4, 4, 2, 2, 3, 3, 3, 4, 3, 4],
             [t(0,0), t(2,45),t(5,41),t(7,14),t(8,24),t(12,20),t(20,6),t(20,44),
              t(21,39),t(24,50),t(24,50),t(27,31),t(29,23),t(31,1),t(31,51)]],
    'oct2': [[0,0],
            [t(0,0), t(35,0)]],
    'oct3': [[2, 2],
             [t(13, 14), t(34, 34)]],
    'oct4': [[0,2,2,0],
            [t(0,0),t(8,23), t(13, 55),t(35,0)]],
    'oct5': [[3, 2, 1, 4],
             [t(3,14),t(12,29), t(13,12), t(27,8)]],
    'oct6': [[1, 1, 3, 1, 4, 2, 1, 1],
             [t(1,56),t(3,51), t(7,15), t(8,15), t(9,7), t(12,16), t(19,26), t(29,19)]]}




plt.plot([x / 60 for x in octo_dct['oct1'][1]], octo_dct['oct1'][0], label='oct1')
plt.plot([x / 60 for x in octo_dct['oct2'][1]], octo_dct['oct2'][0], label='oct2')
plt.plot([x / 60 for x in octo_dct['oct3'][1]], octo_dct['oct3'][0], label='oct3')
plt.plot([x / 60 for x in octo_dct['oct4'][1]], octo_dct['oct4'][0], label='oct4')
plt.plot([x / 60 for x in octo_dct['oct5'][1]], octo_dct['oct5'][0], label='oct5')
plt.plot([x / 60 for x in octo_dct['oct6'][1]], octo_dct['oct6'][0], label='oct6')

plt.legend()
plt.xlabel('Time (minutes)')
plt.ylabel('Activity')
plt.yticks(range(0,4))
plt.title('Octopus Behaviour Video 1')

plt.show()