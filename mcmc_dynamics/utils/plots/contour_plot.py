import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d
import logging

def get_kde_contours(x, y, xlim=None, ylim=None, return_kernel=False):
    if xlim is None and ylim is None:
        xmin, xmax = -4, 4
        ymin, ymax = -4, 4
    else:
        xmin, xmax = xlim[0], xlim[1]
        ymin, ymax = ylim[0], ylim[1]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    if return_kernel:
        return xx, yy, f, kernel

    return xx, yy, f

def get_mapping_significance(kernel, significances):
    """Given a KDE kernel and a number of significances (e.g 0.8), 
    the function returns the values above which 80% of the KDE lies.
    
    """
    
    def compute_integral(kernel, iso):
        sample = kernel.resample(size=5000)
        insample = kernel(sample) > iso
        return insample.sum() / float(insample.shape[0])
    
    def get_step(deltai):
        if deltai < 0.01:
            step = 3.5
        elif 0.01 < deltai < 0.04:
            step = 1
        else:
            step = 0.5
        return 1 + step
    
    integral_inside = [1]
    isos = [0]
    iso = 0.001
    eps = -0.10 #0.01
    
    while min(integral_inside) > min(significances) + eps:
        integral = compute_integral(kernel, iso)
        deltai = np.abs(integral-integral_inside[-1])
        step = get_step(deltai)
        if integral < 0.98: step = max(1.1, step-0.3)
        
        integral_inside.append(integral)
        isos.append(iso)
        iso = iso * step
        logging.debug("int. = {}, delta = {:.4f}, step = {:.4f}, next iso = {:.4f}".format(integral, deltai, step, iso))
    
    interp = interp1d(integral_inside, isos)
    
    return interp(significances)


def test_plot():
    from matplotlib import pyplot as plt

    fig, sub = plt.subplots(1,1, figsize=(8,8))
    sub.set_aspect('equal')
    sub.set_xlim(-3, 3)
    sub.set_ylim(-3, 3)
    
    # contours will be calculated for regions containing 
    # 0.95, 0.87, ... of the samples according to a KDE
    sigma_levels = [0.95, 0.87, 0.68, 0.5]
    
    # each data set should get its own color
    cmaps = ["Blues", "Reds", "Greens", "Purples"]
    
    for idx in range(4):
        # fake data
        c1 = np.random.rand() / 10
        x, y = np.random.multivariate_normal(mean=[3*(np.random.rand()-0.5)*2, 1.5*(np.random.rand()-0.5)*2], cov=[[.2,0], [0, .2]], size=1000).T
        
        xx, yy, f, kernel = get_kde_contours(x, y, return_kernel=True)
        levels = list(get_mapping_significance(kernel, sigma_levels))
        
        sub.contourf(xx, yy, f, cmap=cmaps[idx], levels=levels, alpha=0.6, extend='max')
        

    # lines for orientation at x=0 and y=0
    color_grey = "#CCCCCC"
    sub.hlines(0, -3, 3, colors=color_grey, lw=0.5)
    sub.vlines(0, -3, 3, colors=color_grey, lw=0.5)
    
    fig.tight_layout()
    fig.savefig("test.png")

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_plot()
    
