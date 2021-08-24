import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import cm


def dual_half_circle(center, radius, angle=0, ax=None, colors=('w','k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        #ax.add_artist(wedge)
        ax.add_patch(wedge)
    return [w1, w2]

def test_dual_half_circle_main():
    fig, ax = plt.subplots()
    dual_half_circle((0.5, 0.5), radius=0.3, angle=90, ax=ax)
    ax.axis('equal')
    plt.show()

def plot_multilabel_scatter(X, Y, cmap=cm.get_cmap('tab20'), edgecolor='k',
                            linewidth=0.4, title=None, fig=None, ax=None,
                            radius_scaler=20.0, **kwargs):
    X_std = X.std(axis=0)
    if X.shape[1] > 2:
        biggest_variance = np.argsort(X_std)[-2:]
        X_std = X_std[biggest_variance]
        X = X[:,biggest_variance]

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    n_classes = Y.shape[1]

    radius = ((X_max - X_min)/radius_scaler)[:2].min()
    #radius = (X.max() - X.min())/radius_scaler

    if fig is None:
        fig = plt.figure(figsize=(4, 3))
    if ax is None:
        ax = fig.add_subplot(111)

    for x, y in zip(X, Y):
        theta2s = np.cumsum(np.true_divide(y, y.sum())*360.0)
        theta1 = 0
        if np.isfinite(theta2s[0]):
            for i, theta2 in enumerate(theta2s):
                if theta1 != theta2:
                    w = Wedge(x[:2], radius, theta1, theta2, ec=edgecolor, lw=linewidth,
                              fc=cmap(i), **kwargs)
                    ax.add_patch(w)
                    theta1 = theta2
        else:
            # Not belong to any class
            print('Do not belong to any class')
            w = Wedge(x[:2], radius, 0, 360, ec='black', lw=linewidth,
                      fc='white', **kwargs)
            ax.add_patch(w)

    ax.set_xlim([X_min[0]-X_std[0], X_max[0]+X_std[0]])
    ax.set_ylim([X_min[1]-X_std[1], X_max[1]+X_std[1]])
    ax.axis('equal')
    if title is not None:
        ax.set_title(title)
    return fig
