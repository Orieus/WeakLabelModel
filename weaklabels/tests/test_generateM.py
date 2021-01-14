from weaklabels.WLweakener import generateM
import numpy as np


def testM(c, method, alpha, beta):

    print(f'-- Testing {method}')
    M = generateM(c, model_class=method, alpha=alpha, beta=beta)
    print(np.round(M * 1000) / 1000)
    print(np.sum(M, 0))

    return M


# Common parameters
c = 3

# Tests
method = 'noisy'
print(f'-- Testing {method}')
alpha = [0.1, 0.2, 0.3]
beta = None

method = 'random_noise'
alpha = [0.1, 0.2, 0.3]
beta = [0.2, 0.3, 0.4]
M = testM(c, method, alpha, beta)

alpha = [0.1, 0.2, 0.3]
beta = 0.2
method = 'random_weak'
M = testM(c, method, alpha, beta)

# IPL vs IPL3
alpha = 0
beta = [0.1, 0.2, 0.3]
method = 'IPL'
M = testM(c, method, alpha, beta)

# IPL3
method = 'IPL3'
M = testM(c, method, alpha, beta)

alpha = [0.9, 0.9, 0.9, 0.9]
beta = 0.1
method = 'quasi-IPL'
M = testM(c, method, alpha, beta)

