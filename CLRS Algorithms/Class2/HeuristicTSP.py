from tkinter import *
from random import *
from math import sqrt
import numpy as np

def dist(a, b):
  return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

def nearest_neighbour_algorithm(points):
  global totaldist
  if len(points) == 0: return []
  start = points[0]
  current = points[0]
  nnpoints = [current]
  points.remove(current)
  while len(points) > 0:
    print(len(points), totaldist)
    next = points[0]
    for point in points:
      if dist(current, point) < dist(current, next):
        next = point
    totaldist += dist(current, next)
    nnpoints.append(next)
    if len(points) == 1:
      lasttrip = dist(next, start)
      print(lasttrip)
      print(totaldist)
      print(totaldist + lasttrip)
    points.remove(next)
    current = next
  return nnpoints

#246633474
#246633507
#246646391

#1203406
#1203405


global totaldist
points = []
totaldist = 0


data = np.loadtxt('nn.txt')
data = data[:, 1:3]

points = []
for i in range(33708):
    points.append((data[i, 0], data[i, 1]))

nearest_neighbour_algorithm(points)

