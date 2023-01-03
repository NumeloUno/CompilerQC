import numpy as np
from matplotlib import pyplot as plt
from CompilerQC import paths


def visualize_delta(delta):
    """visualize the effect of delta in the temperature schedule"""
    t, temperature = 1, [1]
    for i in range(100):
        t = t / (1 + t * np.log(1 + delta))
        temperature.append(t)
    return temperature


for delta in [0.01, 0.1, 10]:
    plt.plot(visualize_delta(delta), label=r"$\delta$" + f" = {delta}")
    plt.text(x=40, y=0.9, s=r"$\sigma_{c_k} = const$")
plt.ylabel("temperature")
plt.xlabel("step k")
plt.title("Effect of delta")
plt.legend()
plt.savefig(paths.plots / "effect_of_delta.png")
