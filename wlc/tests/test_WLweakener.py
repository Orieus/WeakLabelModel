import unittest

import numpy as np

from wlc.WLweakener import computeM
from wlc.WLweakener import computeVirtual

def nan_equal(a,b):
    try:
	np.testing.assert_equal(a,b)
    except AssertionError:
	return False
    return True

class TestWLweakener(unittest.TestCase):

    def test_computeM(self):
        M = computeM(c=3, method='supervised')
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert(np.array_equal(M, expected))

    def test_computeVirtual(self):
        z = np.array([0, 1, 2, 3])
        z_bin = computeVirtual(z, c=2, method='IPL')
        expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert(np.array_equal(z_bin, expected))

        z_bin = computeVirtual(z, c=2, method='quasi_IPL')
        expected = np.array([[.5, .5], [0, 1], [1, 0], [np.nan, np.nan]])
        assert(nan_equal(z_bin, expected))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
