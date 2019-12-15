import sys
import numpy as np
import math
from scipy import constants as consts
from scipy import special
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def main(argv: list) -> int:

    xmin = ymin = -100
    xmax = ymax = 100
    x = np.linspace(xmin, xmax, 2000)
    y = np.linspace(ymin, ymax, 2000)
    x, y = np.meshgrid(x, y)

    nlm = [(1, 0, 0), (2, 0, 0), (2, 1, 0)]

    for n, l, m in nlm:
        psi = get_psi(n, l, m)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(np.abs(psi(*ctos(x, y, 0))) ** 2, cmap="gist_heat", extent=[xmin, xmax, ymin, ymax])
        plt.colorbar(im, fraction=0.046, pad=0.03)
        plt.title(f"<{n}, {l}, {m} | {n}, {l}, {m}>")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"{n}{l}{m}.png")
        plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
