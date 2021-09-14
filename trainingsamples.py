import matplotlib.pyplot as plt
import numpy as np

x = [9, 11, 13,15,17,19]
y = [82.34,	82.08,	81.52,	80.60,	79.89,	78.28]

y1 = [93.99, 92.91,	91.89,	90.88,	89.74,	88.65]
y2 = [95.23, 95.19,	94.99,	94.68,93.98,93.51]
y3 = [96.31,96.87,96.73,96.64,96.57,96.75]
plt.xlabel("The input size ")
plt.ylabel("OA(%)")
plt.plot(x,y, label='IndianPines', marker='o' )
plt.plot(x,y1, label='PaviaU', marker='s',)
plt.plot(x,y2, label='Houston2013',marker='^', )
plt.plot(x,y3, label='Xiongan', marker='v', )
plt.legend()
plt.show()