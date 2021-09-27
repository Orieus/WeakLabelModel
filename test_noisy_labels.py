import numpy as np
import weaklabels.WLweakener as wlw
from sklearn.preprocessing import label_binarize

np.random.seed(0)

c = 3
y = np.array([0, 1, 2])
y_bin = label_binarize(y, classes=range(c))


methods_M = ['IPL', 'noisy', 'quasi-IPL', 'random_noise', 'random_weak']
shapes = {'IPL': (2**c, c), 'noisy': (c, c), 'quasi-IPL': (2**c, c),
          'random_noise': (c, c), 'random_weak': (2**c, c)}

# method_assumptions = ['IPL', 'quasi_IPL', 'Mproper']  # problem with Mproper
a = 0.3
b = 0.3

for method in methods_M:
    print(f'###########################################')
    print(f'# ANALYZING M from {method} (a={a}, b={b})')
    print(f'###########################################')
    print(f'True labels are {y}')
    print('True binary labels are')
    print(y_bin)

    wlm = wlw.WLmodel(c, model_class=method)
    M = wlm.generateM(alpha=a, beta=b)
    # M = wlm.computeM(alpha=1-a, beta=b)
    print(f'M shape = {M.shape}')
    assert(M.shape == shapes[method])
    print(np.round(M, 2))

    z = wlm.generateWeak(y)
    print(f'Weak labels are {z}')
    # print('Weak labels in binary from the old version')
    # z_bin = wlw.computeVirtual(z, c, method='IPL')
    # print(z_bin)
    print('Weak labels in binary are')
    z_bin = wlw.binarizeWeakLabels(z, c)
    print(z_bin)

    # for method_a in method_assumptions:
    print('---------------------------------------------')
    # print(f'- Virtual labels assuming M from {method_a} are')
    print(f'- Virtual labels assuming M are')
    v_bin = wlm.virtual_labels_from_M(z)
    print(np.round(v_bin, 2))
    print('---------------------------------------------\n')
