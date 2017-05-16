import numpy as np
import wlc.WLweakener as wlw
from sklearn.preprocessing import label_binarize

np.random.seed(0)

c = 3
y = np.array([0,1,2])
y_bin = label_binarize(y, range(c))

methods_M = ['IPL', 'noisy', 'quasi_IPL', 'random_noise', 'random_weak']
shapes = {'IPL':(2**c, c), 'noisy':(c, c), 'quasi_IPL': (2**c, c),
          'random_noise': (c, c), 'random_weak': (2**c, c)}

method_assumptions = ['IPL', 'quasi_IPL', 'Mproper'] # problem with Mproper

a = 0.7
b = 0.3

for method in methods_M:
    print('###########################################')
    print('# ANALYZING M from {} (a={}, b={})'.format(method, a, b))
    print('###########################################')
    print('True labels are {}'.format(y))
    print('True binary labels are')
    print(y_bin)
    M = wlw.computeM(c, alpha=a, beta=b, method=method)
    print('M shape = {}'.format(M.shape))
    assert(M.shape == shapes[method])
    print(np.round(M,2))
    z = wlw.generateWeak(y, M)
    print('Weak labels are {}'.format(z))
    #print('Weak labels in binary from the old version')
    #z_bin = wlw.computeVirtual(z, c, method='IPL')
    #print(z_bin)
    print('Weak labels in binary are')
    z_bin = wlw.binarizeWeakLabels(z, c)
    print(z_bin)
    for method_a in method_assumptions:
        print('---------------------------------------------')
        print('- Virtual labels assuming M from {} are'.format(method_a))
        v_bin = wlw.computeVirtual(z, c, method=method_a, M=M)
        print(np.round(v_bin, 2))
    print('---------------------------------------------\n')
