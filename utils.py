
import numpy as np

def ctos(x, y, z):
    """ Cartesian to Spherical Coordinates
        :return: r, theta, phi
                where r is in [0, inf), theta in [0, pi] and phi in [0, 2*pi)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r, np.arccos(z / r), np.arctan(y / x)


def stoc(r, theta, phi):
    """ Cartesian to Spherical Coordinates
        :return: x, y, z
        r must be in [0, inf), theta in [0, pi] and phi in [0, 2*pi).
    """
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)