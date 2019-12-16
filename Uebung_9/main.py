import sys
import numpy as np
import math
from scipy import constants as consts
from scipy import special
from matplotlib import pyplot as plt

from utils import *


def get_psi(n, l, m):
    Z = 1
    # hbar = epsilon_0 = electron mass = elementary charge = 1
    a_0 = 4 * np.pi
    # a_0 = 4 * np.pi * consts.epsilon_0 * consts.hbar ** 2 / (consts.electron_mass * consts.elementary_charge ** 2)

    def R_nl(r):
        rho = 2 * Z * r / (n * a_0)
        return np.sqrt((2 * Z / (n * a_0)) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l))) * \
               np.exp(-rho / 2) * rho ** l * special.assoc_laguerre(rho, n - l - 1, 2 * l + 1)

    def psi(r, theta, phi):
        return R_nl(r) * special.sph_harm(m, l, theta, phi)

    return psi

s1 = get_psi(1, 0, 0)
s2 = get_psi(2, 0, 0)
spz = get_psi(2, 1, 0)


def E(n):
    return n + 0.5


def U(t, n):
    return np.exp(-1j * E(n) * t)


def Mt(r, theta, phi, t):
    _s1 = s1(r, theta, phi)
    _s2 = s2(r, theta, phi)

    return 1 / np.sqrt(2) * (np.exp(-1j * E(1) * t) * _s1 + np.exp(-1j * E(2) * t) * _s2)


def Dt(r, theta, phi, t):
    _s1 = s1(r, theta, phi)
    _spz = spz(r, theta, phi)

    return 1 / np.sqrt(2) * (np.exp(-1j * E(1) * t) * _s1 + np.exp(-1j * E(2) * t) * _spz)



def main(argv: list) -> int:

    N = 900

    xmin = ymin = -100
    xmax = ymax = 100
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    x, y = np.meshgrid(x, y)

    nlm = [(1, 0, 0), (2, 0, 0), (2, 1, 0)]

    for n, l, m in nlm:
        psi = get_psi(n, l, m)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(np.abs(psi(*ctos(x, y, 0))) ** 2, cmap="gist_heat", extent=[xmin, xmax, ymin, ymax], origin="lower")
        plt.colorbar(im, fraction=0.046, pad=0.03)
        plt.title(f"<{n}, {l}, {m} | {n}, {l}, {m}>")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"{n}{l}{m}.png")
        # plt.show()
        plt.clf()
        plt.close()

    xmin = ymin = -50
    xmax = ymax = 50
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    x, y = np.meshgrid(x, y)

    for i, t in enumerate(np.linspace(0, 8, 600), start=1):
        fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)

        axes[0].imshow(np.abs(Mt(*ctos(x, y, 0), t * np.pi)) ** 2, cmap="gist_heat", extent=[xmin, xmax, ymin, ymax], origin="lower", vmin=0, vmax=1.1e-4)
        axes[0].set_title("$|<x|M>|^2$")
        axes[1].imshow(np.abs(Dt(*ctos(x, y, 0), t * np.pi)) ** 2, cmap="gist_heat", extent=[xmin, xmax, ymin, ymax], origin="lower", vmin=0, vmax=1.1e-4)
        axes[1].set_title("$|<x|D>|^2$")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.suptitle(f"t = {round(t, 2)} $\pi$")
        plt.savefig(f"images/{i}.png")
        # plt.show()
        plt.clf()
        plt.close()


    return 0


if __name__ == "__main__":
    main(sys.argv)
