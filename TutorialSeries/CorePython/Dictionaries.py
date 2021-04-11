import random
import sys
import os

'{} for Dictionaries'

super_villians = {'Fiddler' : 'Isaac Brown',
                  'Captain Cold' : 'Leonard Snart',
                  'Weather Wizard' : 'Mark Mardon',
                  'Mirror Master' : 'Sam Scudder',
                  'Pied Piper' : 'Thomas Peterson'}

print(super_villians['Captain Cold'])

super_villians['New Villian'] = 'Evil Guy'

del super_villians['Fiddler']

super_villians['Pied Piper'] = 'Hartley Rathaway'

print(len(super_villians))

print(super_villians.get("Pied Piper"))

print(super_villians.keys())

print(super_villians.values())


dict = {"1": [2,5]}

print(dict["1"])

