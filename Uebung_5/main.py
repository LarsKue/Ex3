import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2


@np.vectorize
def phase(z):
    if z == 0:
        return None
    x = np.real(z)
    y = np.imag(z)

    if x < 0 and y == 0:
        return np.pi
    return 2 * np.arctan(y / (np.abs(z) + x))


def phi(x, sx, phi0):
    return phi0 * np.exp(-x ** 2 / (4 * sx))


def phit(k, t, sk, phi0, m):
    # hbar = 1
    w = k ** 2 / (2 * m)
    return phi(k, sk, phi0) * np.exp(-1j * w * t)


def phixt(x, t, sk, phi0, m):
    # hbar = 1
    return phi0 * np.exp(x ** 2 / (4 * (1 / (4 * sk) + 1j * t / (2 * m)) ** 2)) / np.sqrt(2)


def sigmat(t, sk, m):
    # hbar = 1
    return np.sqrt(2) * (1 / (4 * sk) + 1j * t / (2 * m))


def main(argv: list) -> int:
    sx = 1
    sk = 1 / (4 * sx)

    phi0 = 1 / (2 * np.pi * sx) ** (1 / 4)
    phik0 = np.sqrt(2 * sx)

    m = 1  # normed mass

    k = np.linspace(-3, 3, 10000)

    fig = plt.figure(figsize=(12, 12))

    axes = []

    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i + 1))
        axes[i].set_ylim(-1.2, 1.6)
        axes[i].set_xlabel("k")

    for t in range(0, 6, 2):
        psi = phit(k, t, sk, phik0, m)
        axes[0].plot(k, np.abs(psi), label=f"t = {t}")
        axes[0].set_title("Betrag")
        axes[1].plot(k, phase(psi), label=f"t = {t}")
        axes[1].set_title("Phase")
        axes[2].plot(k, np.real(psi), label=f"t = {t}")
        axes[2].set_title("Realteil")
        axes[3].plot(k, np.imag(psi), label=f"t = {t}")
        axes[3].set_title("Imaginärteil")

    for i in range(4):
        axes[i].legend()

    fig.suptitle("Wellenfunktion in Impulsdarstellung zu verschiedenen Zeiten")
    plt.savefig("1c.png", format="png")
    plt.show()


    # for t in range(0, 6, 2):
    #     psi = phit(k, t, sk, phik0, m)
    #     plt.plot(k, np.abs(psi) ** 2, label=f"t = {t}")
    #
    # plt.legend()
    # plt.savefig("1c2.png", format="png")
    # plt.show()



    fig = plt.figure(figsize=(12, 12))
    axes = []

    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i + 1))
        axes[i].set_xlabel("t")

    t = np.linspace(0, 5, 10000)

    sig = sigmat(t, sk, m)

    axes[0].plot(t, np.abs(sig))
    axes[0].set_title("Betrag")
    axes[1].plot(t, phase(sig))
    axes[1].set_title("Phase")
    axes[2].plot(t, np.real(sig))
    axes[2].set_title("Realteil")
    axes[3].plot(t, np.imag(sig))
    axes[3].set_title("Imaginärteil")

    fig.suptitle(r"Wahrscheinlichkeitsbreite $\sigma$ als Funktion der Zeit")
    plt.savefig("1e")
    plt.show()

    fig = plt.figure(figsize=(12, 12))
    axes = []

    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i + 1))
        axes[i].set_xlabel("x")

    x = np.linspace(-3, 3, 10000)

    for t in range(0, 6, 2):
        psi = phixt(x, t, sk, phik0, m)
        axes[0].plot(x, np.abs(psi), label=f"t = {t}")
        axes[0].set_title("Betrag")
        axes[1].plot(x, phase(psi), label=f"t = {t}")
        axes[1].set_title("Phase")
        axes[2].plot(x, np.real(psi), label=f"t = {t}")
        axes[2].set_title("Realteil")
        axes[3].plot(x, np.imag(psi), label=f"t = {t}")
        axes[3].set_title("Imaginärteil")

    for i in range(4):
        axes[i].legend()

    fig.suptitle("Wellenfunktion in Ortsdarstellung zu verschiedenen Zeiten")
    plt.savefig("1f.png", format="png")
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
