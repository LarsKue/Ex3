
import sys

import math
from matplotlib import pyplot as plt
import numpy as np
from scipy import constants as consts
from scipy.special import hermite
from scipy.integrate import quad

from utils import *

consts.hbar = 1


def H(n, x):
    if n == 0:
        return 1
    if n == 1:
        return 2 * x
    if n == 2:
        return 4 * x ** 2 - 2
    if n == 3:
        return 8 * x ** 3 - 12 * x
    if n == 4:
        return 16 * x ** 4 - 48 * x ** 2 + 12
    raise NotImplemented("Please use scipy.special.hermite for greater ns if your argument is real.")


def psi(x, n, m, w):
    return (m * w / (np.pi * consts.hbar)) ** (1 / 4) * 1 / np.sqrt(2 ** 2 * math.factorial(n)) * H(n, np.lib.scimath.sqrt(m * w * x / consts.hbar)) * np.exp(-m * w * x ** 2 / (2 * consts.hbar))


def sp_xD(x, t, m, w):
    return np.exp(-1j * t * w / 2) / np.sqrt(2) * psi(x, 0, m, w) + np.exp(-3j * t * w / 2) / np.sqrt(2) * psi(x, 1, m, w)


def sp_xQ(x, t, m, w):
    return np.exp(-1j * t * w / 2) / np.sqrt(2) * psi(x, 0, m, w) + np.exp(-5j * t * w / 2) / np.sqrt(2) * psi(x, 2, m, w)


def EW_x(t, w):
    return (3 + np.sqrt(2) * np.cos(2 * w * t))


def EW_p(t, w):
    return (3j - np.sqrt(2) * np.cos(2 * w * t))


def main(argv: list) -> int:

    x = np.linspace(-4, 4, 5000)
    m = w = 1  # we use normed values

    i = 0
    for t in np.linspace(0, 2, 600):
        y1 = sp_xD(x, t * np.pi, m, w)
        y2 = sp_xQ(x, t * np.pi, m, w)
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(x, np.abs(y1), label="Betrag")
        ax1.plot(x, np.angle(y1), label="Phase")
        ax1.set_ylim(-3.25, 3.25)
        ax1.set_xlabel("x")
        ax1.set_title("D(t)")
        ax1.legend()

        ax2.plot(x, np.abs(y2), label="Betrag")
        ax2.plot(x, np.angle(y2), label="Phase")
        ax2.set_ylim(-3.25, 3.25)
        ax2.set_xlabel("x")
        ax2.set_title("Q(t)")
        ax2.legend()

        fig.suptitle(f"t={round(t, 2)} $\\pi$")
        # plt.savefig(f"images/t{round(t, 2)}.png")

        plt.savefig(f"images/{i}.png")
        print(i)
        i += 1

        plt.clf()
        plt.close()

    return 0


if __name__ == "__main__":
    main(sys.argv)