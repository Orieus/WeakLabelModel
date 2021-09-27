import unittest

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

# from weaklabels.WLweakener import (
#     computeM, generateM, WLmodel, binarizeWeakLabels)
from WLweakener import (computeM, generateM, WLmodel, binarizeWeakLabels)

from sklearn.preprocessing import label_binarize


class TestWLweakener(unittest.TestCase):

    def test_wlmodel(self):
        wm = WLmodel(c=3, model_class='supervised')
        M = wm.computeM()
        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        assert_array_equal(M, expected)

    def test_computeM(self):
        M = computeM(c=3, model_class='supervised')
        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        assert_array_equal(M, expected)

        M = computeM(c=3, model_class='noisy', alpha=0.8)
        expected = np.array([[.8, .1, .1],
                             [.1, .8, .1],
                             [.1, .1, .8]])
        assert_array_almost_equal(M, expected)

        M = computeM(c=4, model_class='noisy', alpha=0.1)
        expected = np.array([[.1, .3, .3, .3],
                             [.3, .1, .3, .3],
                             [.3, .3, .1, .3],
                             [.3, .3, .3, .1]])
        assert_array_almost_equal(M, expected)

        M = computeM(c=2, model_class='quasi-IPL', beta=0.2)
        # TODO Check if this is the expected M
        expected = np.array([[0., 1.],
                             [1., 0.], ])
        assert_array_equal(M, expected)

        M = computeM(c=3, model_class='quasi-IPL', beta=0.0)
        # TODO Check if this is the expected M
        expected = np.array([[0., 0., 1.],
                             [0., 1., 0.],
                             [0., 0., 0.],
                             [1., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.]])
        assert_array_equal(M, expected)

    # def test_computeVirtual(self):
    #    z = np.array([0, 1, 2, 3])
    #    wm = WLmodel(c=2, model_class='IPL')
    #    z_bin = virtual_labels(z, method='IPL', p=None):
    #    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #    assert_array_equal(z_bin, expected)

    #    z_bin = computeVirtual(z, c=2, method='quasi-IPL')
    #    expected = np.array([[.5, .5], [0, 1], [1, 0], [np.nan, np.nan]])
    #    assert_array_almost_equal(z_bin, expected)

    def test_generateWeak(self):
        c = 4
        y = np.array([0, 1, 2, 3])
        wm = WLmodel(c=c, model_class='supervised')
        M = wm.generateM()
        z = wm.generateWeak(y)
        print("Aqui ha fallao")

        expected = np.array([8, 4, 2, 1])
        assert_equal(z, expected)

    def test_binarizeWeakLabels(self):
        c = 4
        z = np.array([8, 4, 2, 1])
        z_bin = binarizeWeakLabels(z, c)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             ])
        assert_equal(z_bin, expected)

    def test_not_mixing_virtual(self):
        alpha = 0.0
        # TODO Is beta necessary in this test?
        beta = 1.0
        n_classes = 5

        y = np.random.randint(low=0, high=n_classes, size=200)
        # Convert y into a binary matrix
        y_bin = label_binarize(y, list(range(n_classes)))
        y_bin_float = y_bin.astype(float)

        # TODO This creates a NaN error: 'quasi-IPL', 'IPL'
        for mixing_method in ['supervised', 'noisy', 'random_noise',
                              'random_weak']:
            # Generate weak labels
            wm = WLmodel(c=n_classes, model_class=mixing_method)
            M = wm.generateM(alpha=alpha, beta=beta)
            z = wm.generateWeak(y)

            z_bin = binarizeWeakLabels(z, n_classes)
            np.testing.assert_equal(y_bin_float, z_bin)

            # Compute the virtual labels
            # TODO Should these pass the test? 'known-M-opt',
            #                                  'known-M-opt-conv'
            # v_methods = ['IPL', 'quasi-IPL', 'known-M-pseudo']
            # for v_method in v_methods:
            #     virtual = computeVirtual(z, n_classes, method=v_method, M=M)
            #     np.testing.assert_equal(y_bin_float, virtual)

    def test_generateM(self):
        # params in this order: (c, method, alpha, beta, expected)
        params = ((3, 'supervised', None, None,
                   np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])),
                  (3, 'noisy', 0.2, None,
                   np.array([[.8, .1, .1],
                             [.1, .8, .1],
                             [.1, .1, .8]])),
                  (4, 'noisy', 0.9, None,
                   np.array([[.1, .3, .3, .3],
                             [.3, .1, .3, .3],
                             [.3, .3, .1, .3],
                             [.3, .3, .3, .1]])),
                  (4, 'complementary', None, None,
                   np.array([[0, 1 / 3, 1 / 3, 1 / 3],
                             [1 / 3, 0, 1 / 3, 1 / 3],
                             [1 / 3, 1 / 3, 0, 1 / 3],
                             [1 / 3, 1 / 3, 1 / 3, 0]])),
                  (2, 'quasi-IPL', 0.8, 0.2,
                   np.array([[0., 0.],
                             [0., 1.],
                             [1., 0.],
                             [0., 0.]])),
                  (3, 'quasi-IPL', 0.8, 0.0,
                   np.array([[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 1., 0.],
                             [0., 0., 0.],
                             [1., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.]])),
                  # This test fails because of a division by 0 in a matrix B
                  # full of zeros.
                  # (4, 'random_noise', 0.9, np.inf,
                  #  np.array([[.1, .3, .3, .3],
                  #            [.3, .1, .3, .3],
                  #            [.3, .3, .1, .3],
                  #            [.3, .3, .3, .1]])),
                  # This test fails, find reason
                  # (4, 'random_weak', 0.9, np.inf,
                  # np.array([[.1, .3, .3, .3],
                  #           [.3, .1, .3, .3],
                  #           [.3, .3, .1, .3],
                  #           [.3, .3, .3, .1]])),
                  )
        for c, m, a, b, expected in params:
            M = generateM(c=c, model_class=m, alpha=a, beta=b)
            assert_array_almost_equal(
                M, expected, err_msg=(f'c={c}, method={m}, '
                                      f'alpha={a}, beta={b}'))

    def test_generateM_sum_columns(self):
        params = ((3, 'noisy', [0.1, 0.2, 0.3], None),
                  (4, 'complementary', None, None),
                  (3, 'random_noise', [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]),
                  (3, 'random_weak', [0.1, 0.2, 0.3], 0.2),
                  (3, 'IPL', 0, [0.1, 0.2, 0.3]),
                  (3, 'IPL3', 0, [0.1, 0.2, 0.3]),
                  (3, 'quasi-IPL', [0.9, 0.9, 0.9, 0.9], 0.1),
                  )

        for c, method, alpha, beta in params:
            print(f"Method = {method}")
            M = generateM(c, model_class=method, alpha=alpha, beta=beta)
            column_sum = np.sum(M, axis=0)
            expected = np.ones(c)
            assert_array_almost_equal(column_sum, expected)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
