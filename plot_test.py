import matplotlib.pyplot as plt
import numpy as np

x = [10, 20, 30,40,50]
y = [83.08,
88.48,
89.86,
90.52,
90.76]

y1 = [91.71,
92.12,
92.20,
92.26,
92.31]
y2 = [97.53,
98.83,
99.12,
99.32,
99.52]
y3 = [96.36,
97.97,
98.57,
98.58,
98.68]
plt.xlabel("The percentage of used training samples(%) ")
plt.ylabel("OA(%)")
plt.plot(x,y, label='IndianPines', marker='o' )
plt.plot(x,y1, label='PaviaU', marker='s',)
plt.plot(x,y2, label='Houston2013',marker='^', )
plt.plot(x,y3, label='Xiongan', marker='v', )
plt.legend()
plt.show()