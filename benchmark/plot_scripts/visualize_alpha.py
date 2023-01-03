import numpy as np
from matplotlib import pyplot as plt
from CompilerQC import paths


def visualize_alpha(alpha):
    """visualize the effect of delta in the temperature schedule"""
    t, temperature = 1, [1]
    for i in range(100):
        t = alpha * t
        temperature.append(t)
    return temperature


for delta in [0.5, 0.9, 0.99]:
    plt.plot(visualize_alpha(alpha), label=r"$\alpha$" + f" = {alpha}")
    plt.text(x=40, y=0.9, s=r"$t_{k+1} = \alpha t_{k}$")

plt.ylabel("temperature")
plt.xlabel("step k")
plt.title("Effect of alpha")
plt.legend()
plt.savefig(paths.plots / "effect_of_alpha.png")
