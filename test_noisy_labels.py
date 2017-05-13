import numpy as np
import wlc.WLweakener as wlw
from sklearn.preprocessing import label_binarize

np.random.seed(0)

methods = ['IPL', 'noisy', 'quasi_IPL', 'random_noise', 'random_weak']
c = 4
y = np.array([0,1,2,3])
y_bin = label_binarize(y, range(c))

a = 1.0
b = 0.0

for method in methods:
    print('\n#### ANALYZING M from {} ####'.format(method))
    print('True labels are {}'.format(y))
    print('True binary labels are')
    print(y_bin)
    M = wlw.computeM(c, alpha=a, beta=b, method=method)
    z = wlw.generateWeak(y, M, c)
    print('Weak labels are {}'.format(z))
    print('Virtual labels are')
    z_bin = wlw.computeVirtual(z, c, method=method)
    print(z_bin)
    print('Weak labels in binary from the old version')
    z_bin = wlw.computeVirtual(z, c, method='IPL')
    print(z_bin)
    print('Weak labels in binary from the new version')
    z_bin = wlw.binarizeWeakLabels(z, c, method=method)
    print(z_bin)
