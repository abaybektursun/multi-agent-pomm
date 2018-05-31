import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    xar = []
    yar = []
    for x,y in zip(np.arange(50),np.flip(np.arange(50),0)):
        xar.append(int(x))
        yar.append(int(y))
    for x,y in zip(np.arange(50,100),np.arange(50)):
        xar.append(int(x))
        yar.append(int(y))
    ax1.clear()
    ax1.plot(xar,yar)

ani = animation.FuncAnimation(fig, animate, interval=1000)
