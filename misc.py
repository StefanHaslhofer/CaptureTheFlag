import numpy as np
import matplotlib.pyplot as plt
import re

def plot_actions():
    red_tags = 0
    blue_tags = 0
    red_saves = 0
    blue_saves = 0
    red_picks = 0
    blue_picks = 0

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    with open("out.txt", "r") as file:
        for line in file:
            if "TAGGED blue" in line:
                blue_tags += 1
            if "TAGGED red" in line:
                red_tags += 1
            if "FLAG RETURNED by blue" in line:
                blue_saves += 1
            if "FLAG RETURNED by red" in line:
                red_saves += 1
            if "PICKED UP by blue" in line:
                blue_picks += 1
            if "PICKED UP by red" in line:
                red_picks += 1

    ax[0].bar(["Tags", "Saves", "Picks"], [blue_tags, blue_saves, blue_picks])
    ax[0].set_title("Blue Team")

    ax[1].bar(["Tags", "Saves", "Picks"], [red_tags, red_saves, red_picks], color="orange")
    ax[1].set_title("Red Team")

    fig.supxlabel("Actions")
    fig.supylabel("Counts")

    plt.tight_layout()
    plt.show()

def plot_iteration_rewards():
    iteration = 0
    rewards = []

    with open("out.txt", "r") as file:
        for line in file:
            if line.startswith("ITERATION"):
                info = re.split(r'=|:|,| ', line)
                iteration = int(info[1])
                rewards.append(float(info[4]))

    plt.plot(np.linspace(0, iteration, 258), rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Mean reward")
    plt.grid(True)

    plt.show()

def plot_distance_function():
    x = np.linspace(0, 10, 100)

    # two functions
    y1 = 1 / (x + 1)
    y2 = np.full(100, 0.25)

    # plot both on the same axes
    plt.plot(x, y1, label="1/(n+1)")
    plt.plot(x, y2, label="m")

    # extras (highly recommended)
    plt.xlabel("Distance to enemy flag")
    plt.ylabel("|reward|")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    plot_distance_function()
    plot_actions()
    plot_iteration_rewards()