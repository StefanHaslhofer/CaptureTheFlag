import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(0, 10, 100)

    # two functions
    y1 = 1/(x + 1)
    y2 = np.full(100, 0.25)

    # plot both on the same axes
    plt.plot(x, y1, label="1/(n+1)")
    plt.plot(x, y2, label="m")

    # extras (highly recommended)
    plt.xlabel("distance to enemy flag")
    plt.ylabel("|reward|")
    plt.legend()
    plt.grid(True)

    plt.show()