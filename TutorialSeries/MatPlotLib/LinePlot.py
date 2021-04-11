import matplotlib.pyplot as plt


x1 = [0, 1, 2, 3]
x2 = [0, 1, 2, 3]

l1x = [0, 2]
l1y = [0, 3]

l2x = [0, 2.5]
l2y = [0, 3]

l3x = [0, 3]
l3y = [0, 3]

l4x = [0, 3]
l4y = [0, 2.5]

l5x = [0, 3]
l5y = [0, 2]


plt.scatter(x1, x2, color='r', s=200, marker="x")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Hypothesis Function')
plt.xlim(xmin=0)
plt.xlim(xmax=3)
plt.ylim(ymin=0)
plt.ylim(ymax=3)
plt.show()


plt.scatter(x1, x2, color='r', s=200, marker="x")
plt.plot(l1x, l1y, linestyle='solid', color='b')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Hypothesis Function')
plt.xlim(xmin=0)
plt.xlim(xmax=3)
plt.ylim(ymin=0)
plt.ylim(ymax=3)
plt.show()

plt.scatter(x1, x2, color='r', s=200, marker="x")
plt.plot(l1x, l1y, linestyle='solid', color='b')
plt.plot(l2x, l2y, linestyle='solid', color='g')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Hypothesis Function')
plt.xlim(xmin=0)
plt.xlim(xmax=3)
plt.ylim(ymin=0)
plt.ylim(ymax=3)
plt.show()

plt.scatter(x1, x2, color='r', s=200, marker="x")
plt.plot(l1x, l1y, linestyle='solid', color='b')
plt.plot(l2x, l2y, linestyle='solid', color='g')
plt.plot(l3x, l3y, linestyle='solid', color='c')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Hypothesis Function')
plt.xlim(xmin=0)
plt.xlim(xmax=3)
plt.ylim(ymin=0)
plt.ylim(ymax=3)
plt.show()

plt.scatter(x1, x2, color='r', s=200, marker="x")
plt.plot(l1x, l1y, linestyle='solid', color='b')
plt.plot(l2x, l2y, linestyle='solid', color='g')
plt.plot(l3x, l3y, linestyle='solid', color='c')
plt.plot(l4x, l4y, linestyle='solid', color='m')
plt.plot(l5x, l5y, linestyle='solid', color='y')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Hypothesis Function')
plt.xlim(xmin=0)
plt.xlim(xmax=3)
plt.ylim(ymin=0)
plt.ylim(ymax=3)
plt.show()





x1a = 1.5
x2a = 3.5

x1b = 1.25
x2b = 0.875

x1c = 1
x2c = 0

x1d = 0.75
x2d = 0.875

x1e = 0.5
x2e = 3.5


plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()


plt.scatter(x1a, x2a, color='b', s=200, marker="x")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()


plt.scatter(x1a, x2a, color='b', s=200, marker="x")
plt.scatter(x1b, x2b, color='g', s=200, marker="x")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()

plt.scatter(x1a, x2a, color='b', s=200, marker="x")
plt.scatter(x1b, x2b, color='g', s=200, marker="x")
plt.scatter(x1c, x2c, color='c', s=200, marker="x")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()

plt.scatter(x1a, x2a, color='b', s=200, marker="x")
plt.scatter(x1b, x2b, color='g', s=200, marker="x")
plt.scatter(x1c, x2c, color='c', s=200, marker="x")
plt.scatter(x1d, x2d, color='m', s=200, marker="x")
plt.scatter(x1e, x2e, color='y', s=200, marker="x")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()



from scipy.interpolate import spline
import numpy as np

x1 = np.array([0.5, 0.75, 1, 1.25, 1.5])
x2 = np.array([3.5, 0.875, 0, 0.875, 3.5])

xnew = np.linspace(x1.min(),x1.max(),300)
power_smooth = spline(x1,x2,xnew)
plt.plot(xnew,power_smooth)

plt.scatter(x1a, x2a, color='b', s=200, marker="x")
plt.scatter(x1b, x2b, color='g', s=200, marker="x")
plt.scatter(x1c, x2c, color='c', s=200, marker="x")
plt.scatter(x1d, x2d, color='m', s=200, marker="x")
plt.scatter(x1e, x2e, color='y', s=200, marker="x")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Cost Function')
plt.xlim(xmin=0)
plt.xlim(xmax=2)
plt.ylim(ymin=0)
plt.ylim(ymax=4)
plt.show()

# plt.scatter(x1, x2, color='r', s=200, marker="x")
# plt.plot(l1x, l1y, linestyle='solid')
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Hypothesis Function')
# plt.xlim(xmin=0)
# plt.xlim(xmax=3)
# plt.ylim(ymin=0)
# plt.ylim(ymax=3)
# plt.show()










