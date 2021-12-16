import numpy as np
import matplotlib.pyplot as plt
import xylem as xy
import time
import networkx as nx
import pickle
from scipy.spatial.qhull import QhullError
from stats import stat

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20
})

ylabels = {
    'wbridges': 'Fraction of total channel area found in loops',
    'bridges': 'Fraction of total channel length found in loops',
    'nloops': r'Thresholded number of loops per area (km$ ^{-2}$)',
    'loopareas': 'Island area over total area',
    'mstdiff': r'$\Omega$',
    'mstdiffl': 'Minimum fraction of channel length removed to make a tree',
    'resdist': 'Resistance distance from river to ocean',
    'resdist1': 'Resistance distance from tidal nodes to ocean',
    'pathnodes': 'Number of paths from river to each ocean node',
    'flowchange': 'Fraction of thresholded channels that reverse flow',
    'algconn': 'Algebraic connectivity'
}

def jun20rivers(file='wbridges', style='line', thr=1e-4):
    """ simulates and analyses deltas in June 2020.
        The interior of this function can be edited to simulate and process
        the data, but after the data is available in folder jun20rivers,
        the analyze() line is all that is needed
    """

    strengths = np.logspace(-2, 3, 21)
    folder = 'jun20rivers'
    ntot = 35
    def do(s,f):
        """ Simulate a delta with strength s and filename f
            Default arguments are specified in the a = ... line
        """
        a = xy.DeltaNetwork.make_river(s, basin_fraction=0.15,
            density=90, shape='triangle')
        a.simulate()
        a.save(f)
        return a

    def simulate():
        """ Simulates over the range and sample size used
        """
        for n in range(ntot):
            for s in strengths:
                f = folder+'/%0.2f_0.15_90_tri_%d' % (s, n)
                try:
                    a = xy.DeltaNetwork.load(f)
                except FileNotFoundError:
                    try:
                        do(s,f)
                    except ZeroDivisionError:
                        print('.............................Simulation failed')
                        continue

    def fix():
        """ Looks for any simulations that failed and redoes them
        """
        for n in range(ntot):
            for s in strengths:
                f = folder+'/%0.2f_0.15_90_tri_%d' % (s, n)
                try:
                    a = xy.DeltaNetwork.load(f)
                    if np.any(np.isnan(a.C)):
                        print(f)
                        do(s,f)
                except FileNotFoundError:
                    try:
                        do(s,f)
                    except ZeroDivisionError:
                        print('.............................Simulation failed')
                        continue

    def test():
        """ sanity check that fix() worked
        """
        for s in strengths:
            for n in range(ntot):
                try:
                    a = xy.DeltaNetwork.load(
                        folder+'/%0.2f_0.15_90_tri_%d'%(s,n))
                    print(a.C.max())
                except: pass

    def analyze(file='wbridges', style=style, thr=thr):
        """ Produces desired plot using the dataset made here
        """
        stats = {}
        ax = plt.gca()
        try:
            with open(folder+'/'+file+'.p','rb') as f:
                stats = pickle.load(f)
        except FileNotFoundError:
            for s in strengths:
                print(s)
                stats[s] = np.array([])
                for n in range(ntot):
                    try:
                        a = xy.DeltaNetwork.load(
                            folder+'/%0.2f_0.15_90_tri_%d'%(s,n))
                        stats[s] = np.append(stats[s], stat(a, file, thr=thr))
                    except FileNotFoundError:
                        pass
                with open(folder+'/'+file+'.p', 'wb') as f:
                    pickle.dump(stats, f)

        if file == 'nloops':
            for s in strengths: stats[s] *= 10
        cmap = plt.get_cmap('plasma')
        if style == 'line':
            avg = np.array([np.median(stats[s]) for s in strengths])
            firsts = np.array([np.percentile(stats[s], 25) for s in strengths])
            thirds = np.array([np.percentile(stats[s], 75) for s in strengths])
            ax.plot(2*strengths, avg, c=cmap(0), label='Noiseless \nmodel')
            ax.fill_between(2*strengths, firsts, thirds, alpha=0.2,
                color=cmap(0))
            ax.set_xscale('log')
        elif style == 'box':
            stats = [stats[s] for s in strengths]
            labels = ['$10^{-2}$', '', '', '', '$10^{-1}$', '', '', '',
                '$10^{0}$', '', '', '', '$10^{1}$', '', '', '',
                '$10^{2}$', '', '', '', '$10^{3}$']
            plt.boxplot(stats, labels=labels)
        ax.set_xlabel(r'$T^*$')
        return stats

    analyze(file, style)

def riverdomains(file='wbridges'):
    """ Simulates and analyses the deltas simulated on different domain
        shapes. similar structure to big function jun20rivers()
    """
    shapes = ['square', 'sine', 'strip', 'invtriangle']
    folder = 'riverdomains'
    strengths = [0.01, 0.1, 1, 10, 100, 1000]
    ntot = 20
    def fname(s, shape, n):
        """ generates filename string given T* strength s, shape string shape,
            and sample index n
        """
        if shape != 'tri':
            return folder+'/%0.2f_0.15_90_%s_%d' % (s, shape, n)
        elif shape == 'tri':
            return 'jun20rivers/%0.2f_0.15_90_%s_%d' % (s, shape, n)

    def do(s, f, shape):
        """ Simulate a delta with strength s and filename f
            Default arguments are specified in the a = ... line
        """
        a = xy.DeltaNetwork.make_river(s, basin_fraction=0.15,
            density=90, shape=shape)
        a.simulate()
        a.save(f)
        return a

    def simulate():
        """ Simulates over the range and sample size used
        """
        for n in range(ntot):
            for shape in shapes:
                for s in strengths:
                    f = fname(s, shape, n)
                    try:
                        a = xy.DeltaNetwork.load(f)
                    except FileNotFoundError:
                        try:
                            print(f)
                            do(s, f, shape)
                        except ZeroDivisionError:
                            print('.............................Simulation failed')
                            continue

    def fix():
        """ Looks for any simulations that failed and redoes them
        """
        for n in range(ntot):
            for shape in shapes:
                for s in strengths:
                    f = fname(s, shape, n)
                    try:
                        a = xy.DeltaNetwork.load(f)
                        if np.any(np.isnan(a.C)):
                            print(f)
                            do(s, f, shape)
                    except FileNotFoundError:
                        try:
                            print(f)
                            do(s, f, shape)
                        except ZeroDivisionError:
                            print('.............................Simulation failed')
                            continue

    shapes = ['square', 'sine', 'strip', 'invtriangle', 'tri']
    def analyze(file='wbridges', thr=1e-4):
        """ Produces desired plot using the dataset made here
        """
        plt.figure(figsize=(7,8))
        stats = {}
        names = ['Square', 'Sine', 'Strip', 'Inverted\ntriangle', 'Triangle']
        box = []
        if file == 'mstdiff':
            thr = 1e-5
        for i in range(len(shapes)):
            try:
                with open(folder+'/'+file+'_'+shapes[i]+'.p','rb') as f:
                    stats = pickle.load(f)
            except FileNotFoundError:
                for s in strengths:
                    print(s)
                    stats[s] = np.array([])
                    for n in range(ntot):
                        try:
                            a = xy.DeltaNetwork.load(
                                fname(s, shapes[i], n))
                            x = stat(a, file, thr=thr)
                            stats[s] = np.append(stats[s], x)
                        except FileNotFoundError:
                            pass
                    with open(folder+'/'+file+'_'+shapes[i]+'.p', 'wb') as f:
                        pickle.dump(stats, f)
            box.append(stats[1])

            med = [np.median(stats[s]) for s in strengths]
            plt.plot(np.array(strengths)*2, med, label=names[i])

        plt.xscale('log')
        plt.xlabel(r'$T^*$')
        plt.ylabel(ylabels[file])
        plt.legend(title='Domain shape')
        plt.savefig('final/domains/analysis.png', transparent=True, dpi=200)
        plt.savefig('final/domains/analysis.svg', transparent=True)

        plt.figure(figsize=(7,8))
        plt.boxplot(box, labels=names)
        plt.ylabel(ylabels[file])
        plt.savefig('final/domains/box.png', transparent=True, dpi=200)
        plt.savefig('final/domains/box.svg', transparent=True)

        return stats

    analyze(file)

def marshes(file='mstdiff'):
    """ Simulates deltas that have out-of-phase tidal nodes.
    """
    strengths = [0.01, 0.1, 1, 10, 100, 1000]
    noises = [0, 1/3, 2/3, 1]
    folder = 'marshes'
    ntot = 10
    def fname(s, no, n):
        """ generates filename string given T* strength s, relative tidal
            variation (noise) no,
            and sample index n
        """
        if no != 0:
            return folder+'/%0.2f_%0.2f_0.15_90_tri_%d' % (s, no, n)
        elif no == 0:
            return 'jun20rivers/%0.2f_0.15_90_tri_%d' % (s, n)

    def do(s, noise, f, shape='triangle'):
        """ Simulate a delta with strength s and filename f
            Default arguments are specified in the a = ... line
        """
        a = xy.DeltaNetwork.make_marsh(s, noise, basin_fraction=0.15,
            density=90, shape=shape)
        a.simulate()
        a.save(f)
        return a

    def simulate():
        """ Simulates over the range and sample size used
        """
        for n in range(ntot):
            for s in strengths:
                for no in noises:
                    f = fname(s, no, n)
                    try:
                        a = xy.DeltaNetwork.load(f)
                    except FileNotFoundError:
                        try:
                            do(s, no, f)
                        except ZeroDivisionError:
                            print('.............................Simulation failed')
                            continue

    def fix():
        """ Looks for any simulations that failed and redoes them
        """
        for n in range(ntot):
            for s in strengths:
                for no in noises:
                    f = fname(s, no, n)
                    try:
                        a = xy.DeltaNetwork.load(f)
                        if np.any(np.isnan(a.C)):
                            print(f)
                            do(s, no, f)
                    except FileNotFoundError:
                        try:
                            do(s, no, f)
                        except ZeroDivisionError:
                            print('.............................Simulation failed')
                            continue

    def analyze(file='mstdiff'):
        """ Produces desired plot using the dataset made here
        """
        strengths = [0.01, 0.1, 1, 10, 100, 1000]
        stats = {}
        cmap = plt.get_cmap('plasma')
        if file == 'mstdiff':
            thr = 1e-5
        for no in noises:
            try:
                with open(folder+'/'+file+'_%0.2f.p' % no,'rb') as f:
                    stats = pickle.load(f)
            except FileNotFoundError:
                for s in strengths:
                    print(s)
                    stats[s] = np.array([])
                    for n in range(ntot):
                        try:
                            a = xy.DeltaNetwork.load(fname(s, no, n))
                            stats[s] = np.append(stats[s], stat(a, file, thr=thr))
                        except FileNotFoundError:
                            pass
                    with open(folder+'/'+file+'_%0.2f.p' % no, 'wb') as f:
                        pickle.dump(stats, f)
            med = [np.median(stats[s]) for s in strengths]
            if no == 1:
                no = 0.999
            plt.plot(np.array(strengths)*2, med, label='%0.2f'%no, c=cmap(no),)
        plt.xscale('log')
        plt.xlabel(r'$T^*$')
        plt.ylabel(ylabels[file])
        plt.legend(title='Noise')
        return stats

    analyze(file=file)

def marshes_overlay(file='mstdiff'):
    """ Overlays a plot of the statistic 'file' to the active plot based on the
        stats from the 66\% noise model
        not used in the paper
    """
    strengths = [0.01, 0.1, 1, 10, 100, 1000]
    folder = 'marshes'
    ntot = 10
    def fname(s, no, n):
        if no != 0:
            return folder+'/%0.2f_%0.2f_0.15_90_tri_%d' % (s, no, n)
        elif no == 0:
            return 'jun20rivers/%0.2f_0.15_90_tri_%d' % (s, n)

    strengths = [0.01, 0.1, 1, 10, 100, 1000]
    stats = {}
    cmap = plt.get_cmap('magma')
    no = 0.66
    try:
        with open(folder+'/'+file+'_%0.2f.p' % no,'rb') as f:
            stats = pickle.load(f)
    except FileNotFoundError:
        for s in strengths:
            print(s)
            stats[s] = np.array([])
            for n in range(ntot):
                try:
                    a = xy.DeltaNetwork.load(fname(s, no, n))
                    stats[s] = np.append(stats[s], stat(a, file, thr=1e-5))
                    print('m')
                except FileNotFoundError:
                    pass
            print(stats[s])
            with open(folder+'/'+file+'_%0.2f.p' % no, 'wb') as f:
                pickle.dump(stats, f)
    print(stats)
    avg = np.array([np.median(stats[s]) for s in strengths])
    firsts = np.array([np.percentile(stats[s], 25) for s in strengths])
    thirds = np.array([np.percentile(stats[s], 75) for s in strengths])
    ax = plt.gca()
    cmap = plt.get_cmap('plasma')
    ax.plot(2*np.array(strengths), avg, c=cmap(0.66),
        label='66\% noise \nmodel')
    ax.fill_between(2*np.array(strengths), firsts, thirds, alpha=0.2,
        color=cmap(0.66))
    ax.set_xscale('log')

# our T* estimates
T = {'St Clair': 1.5E-01,
     'Mississippi': 5.4E-01,
     'Wax': 9.5E-01,
     'Mossy': 5.6E-01,
     'Kolyma': 5.0E-01,
     'Colville': 4.7E-01,
     'Apalachicola': 5.9E+00,
     'Mackenzie': 3.5E+00,
     'Orinoco': 1.4E+01,
     'Yenisei': 2.5E+00,
     'Lena': 2.9E+01,
     'Yukon': 7.5E+00,
     'Betsiboka': 8.1E+01,
     'Irrawaddy': 5.0E+02,
     'GBM': 1.2E+02,
     'Rajang': 2.3E+02,
     'Niger': 3.2E+02,
     'Sarawak': 2.5E+03,
     'Ras Isa': 6.0E+03,
     'Barnstable': 5.5E+03
    }

# Nienhuis T* estimates, with our estimates substituted in where they
# do not provide an estimate
TN = {'St Clair': 0,
     'Mississippi': 2.66E-01,
     'Wax': 0,
     'Mossy': 0,
     'Kolyma': 5.91E-01,
     'Colville': 2.03E-01,
     'Apalachicola': 1.43E-01,
     'Mackenzie': 3.22E+01,
     'Orinoco': 1.19E+00,
     'Yenisei': 1.35E-02,
     'Lena': 4.29E-02,
     'Yukon': 7.4E-01,
     'Betsiboka': 9.90E-01,
     'Irrawaddy': 3.53E+00,
     'GBM': 0,
     'Rajang': 2.5E+00,
     'Niger': 0,
     'Sarawak': 6.23E+00,
     'Ras Isa': 0,
     'Barnstable': 0
    }

def getStat(file='mstdiff'):
    """ calculate statistics and store them (or access them if already stored)
        for the real deltas
    """
    try:
        with open('shp/'+file+'.p', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        x = []
        for d in T.keys():
            x.append(stat(xy.newjson(d), file, thr=1e-4))
            print(d, x[-1])
        with open('shp/'+file+'.p', 'wb') as f:
            pickle.dump(x, f)
        return x

def realdataplot(file='wbridges', style='line'):
    """ Create a figure with the statistics from the real deltas
        in log T* space.
        Also calls jun20rivers() to draw the lines for simulated deltas
    """
    plt.figure(figsize=(8,9))
    jun20rivers(file=file, style=style, thr=1e-5)
    plt.xscale('log')
    plt.ylabel(ylabels[file])
    if file in ['resdist', 'resdist1', 'pathnodes', 'flowchange']:
        return None

    y = getStat(file)
    Tstar = np.array(list(T.values()))
    labels = list(T.keys())
    y = np.array(y)

    print({labels[i]: y[i] for i in range(len(labels))})

    # plot data points with x error bars
    xerr = []
    for d in T.keys():
        if TN[d] == 0:
            xerr.append(T[d])
        else:
            xerr.append(TN[d])
    #plt.errorbar(Tstar, y, c='k', ls='', ms=10,
    #    xerr = [0.5*Tstar, 1.001*Tstar], lw=0.2)
    plt.plot(Tstar, y, 'g.', label='Data', ms=14)

    # adjust data labels
    for i in range(len(Tstar)):
        #These settings for wbridges
        if file == 'wbridges':
            if labels[i] in ['Lena']:
                plt.text(0.65*Tstar[i], y[i]+0.025, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Ras Isa']:
                plt.text(0.4*Tstar[i], y[i]+0.025, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Barnstable']:
                plt.text(0.2*Tstar[i], y[i]-0.025, labels[i], fontsize=14,
                    alpha=0.5)
            else:
                plt.text(0.65*Tstar[i], y[i]+0.005, labels[i], fontsize=14,
                    alpha=0.5)
        #These settings for loopareas
        elif file == 'loopareas':
            if labels[i] in ['Mississippi']:
                plt.text(0.65*Tstar[i], y[i]+0.015, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Ras Isa']:
                plt.text(0.4*Tstar[i], y[i]+0.018, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Barnstable']:
                plt.text(0.2*Tstar[i], y[i]+0.005, labels[i], fontsize=14,
                    alpha=0.5)
            else:
                plt.text(0.65*Tstar[i], y[i]+0.005, labels[i], fontsize=14,
                    alpha=0.5)
        elif file == 'nloops':
            if labels[i] in ['Mississippi']:
                plt.text(0.65*Tstar[i], y[i]+65, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Wax']:
                plt.text(0.85*Tstar[i], y[i]+4, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Mossy', 'Colville', 'Kolyma']:
                plt.text(0.65*Tstar[i], y[i]+74, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Ras Isa']:
                plt.text(0.45*Tstar[i], y[i]+20, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Betsiboka']:
                plt.text(0.65*Tstar[i], y[i]+13, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Barnstable']:
                plt.text(0.2*Tstar[i], y[i]+30, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Sarawak']:
                plt.text(0.4*Tstar[i], y[i]+4, labels[i], fontsize=14,
                    alpha=0.5)
            else:
                plt.text(0.65*Tstar[i], y[i]+4, labels[i], fontsize=14,
                    alpha=0.5)
        #otherwise mstdiff
        else:
            if labels[i] in ['Ras Isa']:
                plt.text(0.35*Tstar[i], y[i]+0.003, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Barnstable']:
                plt.text(0.2*Tstar[i], y[i]+0.0005, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Orinoco']:
                plt.text(0.65*Tstar[i], y[i]+0.0008, labels[i], fontsize=14,
                    alpha=0.5)
            elif labels[i] in ['Yenisei', 'Mackenzie']:
                plt.text(0.5*Tstar[i], y[i]+0.0005, labels[i], fontsize=14,
                    alpha=0.5)
            else:
                plt.text(0.65*Tstar[i], y[i]+0.0005, labels[i], fontsize=14,
                    alpha=0.5)
    plt.tick_params()

    # binning
    bins = np.logspace(-1, 4, 6)
    biny = np.zeros(bins.shape[0] - 1)
    binarg = np.searchsorted(bins, Tstar)
    for i in range(len(y)):
        biny[binarg[i]-1] += y[i]
    biny /= np.unique(binarg, return_counts=True)[1]
    plt.plot(10**((np.log10(bins[:-1])+np.log10(bins[1:]))/2),
        biny, '--', color='g')
    '''for i in range(len(bins)-1):
        plt.plot([bins[i], bins[i+1]], [biny[i], biny[i]], '-',
            color='g', alpha=0.2)
        plt.plot([bins[i], bins[i]], [biny[i]-0.0025, biny[i]+0.0025], '-',
            color='g', alpha=0.2)
        plt.plot([bins[i+1], bins[i+1]], [biny[i]-0.0025, biny[i]+0.0025], '-',
            color='g', alpha=0.2)'''

def align(file='wbridges', plot=False):
    """ Finds the amount that the peak of data measurements and simulated delta
        statistics differ
        not used in paper
    """
    from scipy.interpolate import interp1d

    strengths = np.append([1e-10], np.append(np.logspace(-2, 3, 21), 1e5))
    with open('jun20rivers/'+file+'.p','rb') as f:
        stats = pickle.load(f)
    avg = np.append([0],
        np.append([np.median(stats[s]) for s in strengths[1:-1]], 0))
    with open('shp/'+file+'.p', 'rb') as f:
        y = np.array(pickle.load(f))

    strengths *= 2
    deltasT = np.array(list(T.values()))
    def error(x):
        f = interp1d(x[0]*strengths, x[1]*avg)
        return np.linalg.norm(f(deltasT) - y, ord=2)

    from scipy.optimize import minimize
    res = minimize(error, x0=[1,1])
    if plot:
        plt.plot(res.x[0]*strengths, res.x[1]*avg, '--', color='tab:green')
    print(res.x)

def linearStatsAnalysis(file='mstdiff'):
    """ fits two lines to the left and right sides of the real datapoints when
        their statistic values are shuffled (but T* values kept the same)
        Does this 5e4 times
        Prints what fraction have left slope and right slope at least as strong
        as the real data
        Returns all the slopes
    """
    x = np.log10(list(T.values()))
    y = np.array(getStat(file))

    shuffles = 50000
    peaks = np.arange(6, 15)

    left = np.zeros((shuffles, len(peaks)))
    right = np.zeros((shuffles, len(peaks)))

    for i in range(shuffles):
        if i % 10000 == 0:
            print(i)
        y = np.random.permutation(y)
        for j in range(len(peaks)):
            left[i,j] = np.linalg.lstsq(
                np.stack((np.ones(peaks[j]), x[:peaks[j]]), axis=1),
                y[:peaks[j]], rcond=None)[0][1]
            right[i,j] = np.linalg.lstsq(
                np.stack((np.ones(len(x)-peaks[j]), x[peaks[j]:]), axis=1),
                y[peaks[j]:], rcond=None)[0][1]

    x = np.log10(np.sort(list(T.values())))
    y = np.array(getStat('wbridges'))
    for p in np.arange(6, 15):
        leftslope = np.linalg.lstsq(
            np.stack((np.ones(p), x[:p]), axis=1), y[:p],
            rcond=None)[0][1]
        rightslope = np.linalg.lstsq(
            np.stack((np.ones(len(x)-p), x[p:]), axis=1), y[p:],
            rcond=None)[0][1]
        print(leftslope, rightslope)
        print(np.sum(np.logical_and(
            left[:,p-6] > leftslope, right[:,p-6] < rightslope)))

    return left, right

def quadraticStatsAnalysis(file='mstdiff'):
    """ Randomizes data as above and find what fraction of the 5e4 samples
        have a trend as strong as the data. If the original data has signed
        curvature C and peak at P, the fit to randomized data must have
        peak p at P/2 < p < 2P and curvature greater in magnitude than C with
        the same sign
    """
    x = np.log10(np.sort(list(T.values())))
    y = np.array(getStat(file))

    shuffles = 50000

    # curvature
    c = np.zeros(shuffles)

    # peak location
    p = np.zeros(shuffles)

    for i in range(shuffles):
        if i % 10000 == 0:
            print(i)
        y = np.random.permutation(y)
        sol = np.linalg.lstsq(
            np.stack((np.ones(len(x)), x, x**2), axis=1), y, rcond=None)[0]

        c[i] = sol[2]
        p[i] = -sol[1]/(2*sol[2])

    x = np.log10(list(T.values()))
    y = np.array(getStat('mstdiff'))

    sol = leftslope = np.linalg.lstsq(
        np.stack((np.ones(len(x)), x, x**2), axis=1), y, rcond=None)[0]
    peak = -sol[1]/(2*sol[2])
    print(sol[2], peak)
    print(np.sum((c < sol[2]) * \
        np.logical_and(p > peak/2, p < peak*2)))

    return c, p

def absoluteValueStatsAnalysis(file = 'mstdiff'):
    """ Randomizes data above and checks which of the fits of a function
        a|x-b| + cx + d have a more negative than the true data,
        c more positive than the true data,
        and b between 0 and 2 (so 1 < peak T* < 100)
    """
    from scipy.optimize import curve_fit
    def f(x, a, b, c, d):
        return a*np.abs(x-b) + c*x + d

    x = np.log10(list(T.values()))
    y = np.array(getStat(file))

    opt, _ = curve_fit(f, x, y, p0=[0,0,0,0.05], bounds=(-2, 2))
    print(opt)

    shuffles = 50000

    avec = np.zeros(shuffles)
    bvec = np.zeros(shuffles)
    cvec = np.zeros(shuffles)
    dvec = np.zeros(shuffles)

    for i in range(shuffles):
        y = np.random.permutation(y)
        try:
            (avec[i], bvec[i], cvec[i], dvec[i]), _ = \
                curve_fit(f, x, y, p0=[0,0,0,0.5])
        except RuntimeError:
            avec[i] = np.average(avec[:i])
            bvec[i] = np.average(bvec[:i])
            cvec[i] = np.average(cvec[:i])
            dvec[i] = np.average(dvec[:i])

    print(np.sum((avec < opt[0]) & (cvec > opt[2]) & \
        np.logical_and(bvec > 0, bvec < 2)))

    ax1 =plt.subplot(141)
    plt.hist(avec)
    ax2 = plt.subplot(142)
    plt.hist(bvec)
    ax3 = plt.subplot(143)
    plt.hist(cvec)
    ax4 = plt.subplot(144)
    plt.hist(dvec)
    plt.show()

    return avec, bvec, cvec

if __name__ == '__main__':
    a = xy.DeltaNetwork.load('jun20rivers/0.02_0.15_90_tri_34.p')
    #linearStatsAnalysis()
    #quadraticStatsAnalysis()
    #absoluteValueStatsAnalysis()
