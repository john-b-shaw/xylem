import analyze as an
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import xylem as xy
from stats import stat
import simulate as sim
import time

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20
})

# our T* values
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

# main figures
def riverdisplay2x3(folder='jun20rivers', s=[0.01, 0.1, 1, 10, 100, 1000], mode='c'):
    fig = plt.figure(figsize=(12,10))

    n = 0
    k = 1
    for i in range(len(s)):
        #tide strength
        if mode == 'c':
            if folder[-2:] == 'sq':
                a = xy.DeltaNetwork.load(folder + \
                    '/%0.2f_0.15_90_sq_%d' % (s[i],n))
            else:
                a = xy.DeltaNetwork.load(folder + \
                    '/%0.2f_0.15_90_tri_%d' % (s[i],n))
        #basin density
        elif mode == 'd':
            if folder[-2:] == 'sq':
                a = xy.DeltaNetwork.load(folder + \
                    '/1.00_%0.2f_90_sq_%d' % (s[i],n))
            else:
                a = xy.DeltaNetwork.load(folder + \
                    '/1.00_%0.2f_90_tri_%d' % (s[i],n))
        fig.add_subplot(int(np.ceil(len(s)/3)), 3, k)
        a.plot(drawspecial=False)
        a.plot(style='loops', drawspecial=False)
        if mode == 'c':
            plt.title(r'$T^*=%0.2f$' % s[i], fontsize=20)
        elif mode == 'd':
            plt.title('d=%0.2f' % s[i])
        plt.axis('off')
        k += 1

    plt.tight_layout()
    plt.savefig('lineup2x3.png', transparent=True, dpi=200)

def riverdisplay(folder='jun20rivers', s=[0.01, 0.1, 1, 10, 100, 1000], mode='c'):
    import gdal
    fig = plt.figure(figsize=(20,10))
    grid = GridSpec(2, 2*len(s))

    n = 1
    for i in range(len(s)):
        if folder[-2:] == 'sq':
            a = xy.DeltaNetwork.load(folder + \
                '/%0.2f_0.15_90_sq_%d' % (s[i],n))
        else:
            a = xy.DeltaNetwork.load(folder + \
                '/%0.2f_0.15_90_tri_%d' % (s[i],n))
        fig.add_subplot(grid[0, 2*i:2*i+2])
        a.plot(style='loops', thr=1e-4, drawspecial=False, magn=6)
        plt.title(r'$T^*=%0.2f$' % s[i], fontsize=20)
        plt.axis('off')

    spots = [grid[1, 1:3], grid[1, 3:5], grid[1, 7:9], grid[1, 9:11]]
    deltas = ['Mississippi', 'Orinoco', 'Rajang', 'Barnstaple']

    '''for i in range(len(spots)):
        d = deltas[i]
        dataset = gdal.Open('shp/'+d+'/'+d+'_clipped.tif')
        band = dataset.GetRasterBand(1)
        arr = band.ReadAsArray().T
        scale = dataset.GetGeoTransform()[1]
        xs = arr.shape[0]
        fig.add_subplot(spots[i])
        plt.imshow(arr, cmap='Greys', extent=[0, arr.shape[1]*scale/1000, 0,
            arr.shape[0]*scale/1000],)
        plt.axis('off')
        plt.plot([0.1, 1.1], [0.1,0.1], '-r', lw=4)'''

    plt.tight_layout()
    plt.savefig('lineup.png', dpi=200, transparent=True)#plt.show()

def riverdisplay_vertical(folder='jun20rivers', mode='c'):
    p = 10**np.array([-1.25, -0.25, 0.75, 1.75, 2.75])

    fig = plt.figure(figsize=(3,12))
    grid = GridSpec(len(p), 1)
    n = 0
    for i in range(len(p)):
        if folder[-2:] == 'sq':
            a = xy.DeltaNetwork.load(folder + \
                '/%0.2f_0.15_90_sq_%d' % (p[i],n))
        else:
            a = xy.DeltaNetwork.load(folder + \
                '/%0.2f_0.15_90_tri_%d' % (p[i],n))
        fig.add_subplot(grid[i, 0])
        a.plot(style='loops', thr=1e-5, drawspecial=False, magn=4, c=[0.35,0,0.5])
        plt.text(-0.8, 0.5, r'$10^{%d}$' % np.around(0.25+np.log10(p[i])), fontsize=18)
        plt.axis('off')
        plt.gca().autoscale()
    plt.tight_layout()
    plt.savefig('lineup_vert.png', transparent=False, dpi=200)
    #plt.savefig('lineup_vert.svg', transparent=True, )

def wbridges():
    sim.realdataplot('wbridges', 'line')
    #sim.marshes_overlay('wbridges')
    #plt.savefig('final/main/wbridges.svg',transparent=True,)
    plt.savefig('final/main/wbridges.png',transparent=True, dpi=200)

# sup figures
def initial_conditions():
    """ Set alpha in DeltaNetwork to 0.7 before running this function
    """

    plt.figure(figsize=(8,8))
    ax = plt.gca()
    a = xy.DeltaNetwork.make_river(1, density=90, shape='triangle')
    a.plot(magn=4, ax=ax)
    plt.xlim([-1,1.05]); plt.ylim([-1.05,1])
    plt.xticks([-1,-0.5,0,0.5,1]); plt.yticks([-1,-0.5,0,0.5,1])
    plt.axis('equal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('final/init.png', dpi=200, transparent=True)
    plt.savefig('final/init.svg', transparent=True)

def pressure_ensemble():
    import scipy
    from matplotlib import cm
    from matplotlib.colors import LogNorm, Normalize
    a = xy.DeltaNetwork.load('jun20rivers/1.00_0.15_90_tri_23')
    #a.ensembleplot('final/flowsmovies')

    plt.rcParams.update({
        "font.size": 24
    })

    a.ds = a.fluctuation_ensemble()

    CM = a.C_matrix_sparse(a.C)
    G = a.G_matrix_sparse(CM)[1:,1:].tocsc()
    p = scipy.sparse.linalg.spsolve(G, a.ds[1:,:])
    v = (np.amin(p), np.amax(p)-np.amin(p))
    #p = np.concatenate((p, p[:,::-1]), axis=1)

    t = np.linspace(0, 2*np.pi, p.shape[1])
    tides = np.cos(t)
    t *= p.shape[1]/2/np.pi

    fig = plt.figure(figsize=(16,8))
    grid = plt.GridSpec(1, 101, hspace=0.05, wspace=0.1, left=0.02, right=0.98)

    ax1 = fig.add_subplot(grid[0,0:50])
    a.plot(style='pressure', cmap='cividis', p=p[:,0], v=v,
        drawspecial=False, ax=ax1, magn=6)
    a.plot(style='pipes', drawspecial=False, ax=ax1)
    plt.axis('off')

    ax2 = fig.add_subplot(grid[0,50:])
    a.plot(style='pressure', cmap='cividis', p=p[:,-1], v=v,
        drawspecial=False, ax=ax2, magn=6)
    a.plot(style='pipes', drawspecial=False, ax=ax2)
    plt.axis('off')

    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
        cmap='cividis'), ax=ax2,
        label='Fraction of max potential')

    plt.savefig('final/pressures.png', dpi=100, transparent=True)
    #plt.savefig(dir+'/%05d.png' % i)

def domainsdraw():
    def f(s, shape, n):
        if shape != 'tri':
            return 'riverdomains/%0.2f_0.15_90_%s_%d' % (s, shape, n)
        elif shape == 'tri':
            return 'jun20rivers/%0.2f_0.15_90_%s_%d' % (s, shape, n)

    def load(f):
        return xy.DeltaNetwork.load(f)

    shapes = ['square', 'sine', 'strip', 'invtriangle', 'tri']
    names = ['Square', 'Sine', 'Strip', 'Inverted triangle', 'Triangle']

    fig = plt.figure(figsize=(12,8))
    gs = GridSpec(2,6)
    n = 0
    eps = 0.05

    fig.add_subplot(gs[0,:2])
    load(f(1.00, shapes[0], n)).plot(drawspecial=False)
    plt.title(names[0])
    plt.xlim([-1-eps,1+eps]), plt.ylim([-1-eps,1+eps]), #plt.axis('off')
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])
    plt.yticks(ticks=[-1, -0.5, 0, 0.5, 1])

    fig.add_subplot(gs[0,2:4])
    load(f(1.00, shapes[1], n)).plot(drawspecial=False)
    plt.title(names[1])
    plt.xlim([-1-eps,1+eps]), plt.ylim([-1-eps,1+eps]), #plt.axis('off')
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])
    plt.yticks(ticks=[-1, -0.5, 0, 0.5, 1], labels=['','','','',''])

    fig.add_subplot(gs[0,4:])
    load(f(1.00, shapes[2], n)).plot(drawspecial=False)
    plt.title(names[2])
    plt.xlim([-1-eps,1+eps]), plt.ylim([-1-eps,1+eps]), #plt.axis('off')
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])
    plt.yticks(ticks = [-1, -0.5, 0, 0.5, 1], labels=['','','','',''])

    fig.add_subplot(gs[1,1:3])
    load(f(1.00, shapes[3], n)).plot(drawspecial=False)
    plt.title(names[3])
    plt.xlim([-1-eps,1+eps]), plt.ylim([-1-eps,1+eps]), #plt.axis('off')
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])

    fig.add_subplot(gs[1,3:5])
    load(f(1.00, shapes[4], n)).plot(drawspecial=False)
    plt.title(names[4])
    plt.xlim([-1-eps,1+eps]), plt.ylim([-1-eps,1+eps]), #plt.axis('off')
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])
    plt.yticks(ticks = [-1, -0.5, 0, 0.5, 1], labels=['','','','',''])

    plt.tight_layout()

    plt.savefig('final/domains/domainsdraw.png', transparent=True, dpi=200)
    plt.savefig('final/domains/domainsdraw.svg', transparent=True)

def domainsanalysis():
    sim.riverdomains('mstdiff')

def noise():
    plt.figure(figsize=(7,8))
    sim.marshes()
    plt.savefig('final/noise/noise.png', transparent=True, dpi=200)
    #plt.savefig('final/noise/noise.svg', transparent=True)

    def fname(s, no, n):
        if no != 0:
            return 'marshes'+'/%0.2f_%0.2f_0.15_90_tri_%d' % (s, no, n)
        elif no == 0:
            return 'jun20rivers/%0.2f_0.15_90_tri_%d' % (s, n)
    s = [0.1, 10, 1000]
    no = [0.0, 1/3, 2/3, 1.0]
    no = no[::-1]
    cmap = plt.get_cmap('plasma')
    plt.figure(figsize=(8,10))
    i = 1
    for noi in range(len(no)):
        for si in range(len(s)):
            plt.subplot(len(no), len(s), i)
            a = xy.DeltaNetwork.load(fname(s[si], no[noi], 2))
            a.plot(drawspecial=False, magn=3)
            a.drawloops(c=cmap(no[noi]), thr=1e-5)
            if si == 0:
                plt.text(-1, -0.5, r'noise $=%0.2f$' % no[noi], rotation=90,
                    fontsize=16)
                #plt.ylabel(r'noise $=%0.2f$' % no[noi])
            if noi == 0:
                plt.title(r'$T^*=%0.2f$' % s[si])

            plt.axis('off')
            i += 1
    plt.subplots_adjust(bottom=0.02, top=0.95, left=0.1, right=0.95,
        hspace=0.02, wspace=0.02)
    plt.savefig('final/noise/draw.png', dpi=200, transparent=True)
    #plt.savefig('final/noise/draw.svg', transparent=True)

def entropy():
    try:
        a = xy.DeltaNetwork.load('entropy/delta')
        e = np.load('entropy/entropy.npy')
    except:
        a = xy.DeltaNetwork.make_river(1, density=60, shape='triangle')
        a.simulate(entropy=True, plot=True, movie_dir='entropy')
        a.save('entropy/delta')
        np.save('entropy/entropy.npy', a.entropy)
        e = a.entropy

    plt.figure()
    a.plot()
    plt.axis('off')
    plt.savefig('entropy/final.png', dpi=300)

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(np.arange(e.shape[0])/10, e[:,0], label='Pure river')
    plt.plot(np.arange(e.shape[0])/10, e[:,1], label='Pure tides')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('Graph entropy S/k', fontsize=20)
    plt.legend(fontsize=20)

    plt.subplot(122)
    a.plot()
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('entropy/entropy.png', dpi=200)

def nloops_flowchange():
    plt.rcParams.update({
        "font.size": 24
    })
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    sim.jun20rivers('nloops', style='line', thr=1e-5)
    plt.ylabel('Number of loops in simulation')

    plt.subplot(122)
    sim.jun20rivers('flowchange', style='line', thr=1e-5)
    plt.ylabel('Fraction of flow-reversing edges')
    plt.tight_layout()
    plt.savefig('final/nloops_flowchange.png', transparent=True)
    #plt.savefig('final/nloops_flowchange.svg', transparent=True)

def resdist_npaths(style='line'):
    plt.rcParams.update({
        "font.size": 24
    })
    plt.figure(figsize=(18,6))
    ax1 = plt.subplot(131)
    sim.jun20rivers('resdist', style=style, thr=1e-5)
    plt.ylabel('RD from river to coast')
    plt.yscale('log')

    ax2 = plt.subplot(132, sharey=ax1)
    sim.jun20rivers('resdist1', style=style, thr=1e-5)
    plt.ylabel('RD from tidal nodes to coast')
    plt.yscale('log')

    plt.subplot(133)
    sim.jun20rivers('pathnodes', style=style, thr=1e-5)
    plt.ylabel('River to coast number of paths')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('final/resdist_npaths.png', transparent=True)
    #plt.savefig('final/resdist_npaths.svg', transparent=True)

def convergence():
    plt.figure(figsize=(5,22.5))
    sim.jan20rivers('mstdiff')
    plt.subplots_adjust(left=0.25, hspace=0.25)
    plt.savefig('final/persistence/density_mstdiff.png',transparent=True,
        dpi=200)
    #plt.savefig('final/persistence/density_mstdiff.svg',transparent=True,)

    plt.figure(figsize=(5,22.5))
    sim.jan20rivers('wbridges')
    plt.subplots_adjust(left=0.25, hspace=0.25)
    plt.savefig('final/persistence/density_wbridges.png',transparent=True,
        dpi=200)
    #plt.savefig('final/persistence/density_wbridges.svg',transparent=True,)

    plt.figure(figsize=(5,22.5))
    sim.jan20rivers('nloops')
    plt.subplots_adjust(left=0.25, hspace=0.25)
    plt.savefig('final/persistence/density_nloops.png',transparent=True,
        dpi=200)
    #plt.savefig('final/persistence/density_nloops.svg',transparent=True,)

def backboneprocessing():
    a = xy.DeltaNetwork.load('jun20rivers/1.00_0.15_90_tri_22')
    plt.figure(figsize=(12,8))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    ax = plt.subplot(232)
    a.plot('sticks', thr=1e-3, drawspecial=False)
    plt.axis('off')
    plt.title('Thresholding directly')
    G = a.G.copy()

    plt.subplot(234)
    delattr(a, 'G')
    a.to_networkx(thr=1e-8)
    a.plot('sticks', drawspecial=False)
    plt.axis('off')
    plt.title('1. Set low threshold')

    plt.subplot(235)
    a.remove_trees_nx(1e-8)
    a.plot('sticks', drawspecial=False)
    plt.axis('off')
    plt.title("2. Trim trees (find ''backbone'')")

    plt.subplot(236)
    a.thin(thr=1e-3)
    #cs = nx.get_edge_attributes(a.G, 'conductivity')
    #a.G.remove_edges_from([key for key in cs.keys() if cs[key] < 1e-3])
    #a.G.remove_nodes_from([n[0] for n in a.G.degree if n[1] == 0])
    a.plot('sticks', drawspecial=False)
    plt.axis('off')
    plt.title('3. Threshold backbone')

    G.remove_edges_from(a.G.edges)
    pos = nx.get_node_attributes(G, 'pos')
    #nx.draw_networkx(G, pos, edge_color='r', width=3, ax=ax)
    nodes = list(set([e[0] for e in G.edges]+[e[1] for e in G.edges]))
    nx.draw_networkx(G, pos, nodelist=nodes, with_labels=False,
        edge_color='r', node_color='r', node_size=2, width=2, ax=ax, )

    plt.savefig('backbone.png', dpi=300)
    plt.show()

def nloops_data():
    plt.rcParams.update({
        "font.size": 24
    })
    sim.realdataplot('nloops', 'line')
    plt.tight_layout()
    #plt.savefig('final/stats/nloops.svg',transparent=True,)
    plt.savefig('final/stats/nloops.png',transparent=True, dpi=150)

def loopareas_data():
    plt.rcParams.update({
        "font.size": 24
    })
    sim.realdataplot('loopareas')
    plt.tight_layout()
    plt.savefig('final/stats/loopareas.png', transparent=True, dpi=150)
    #plt.savefig('final/stats/loopareas.svg', transparent=True,)

def persistence(f='mstdiff'):
    ylabels = {
        'wbridges': 'Fraction of total channel area found in loops',
        'bridges': 'Fraction of total channel length found in loops',
        'nloops': r'Thresholded number of loops per area (km$ ^{-2}$)',
        'loopareas': 'Island area over total area',
        #'mstdiff': 'Minimum fraction of channel area removed to make a tree',
        'mstdiff': r'$\Omega$',
        'mstdiffl': 'Minimum fraction of channel length removed to make a tree',
        'resdist': 'Resistance distance from river to ocean',
        'resdist1': 'Resistance distance from tidal nodes to ocean',
        'pathnodes': 'Number of paths from river to each ocean node',
        'flowchange': 'Fraction of thresholded channels that reverse flow',
        'algconn': 'Algebraic connectivity'
    }
    strengths = np.logspace(-2, 3, 21)
    thrs = np.logspace(-5, -2, 30)
    slabels = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ntot = 8
    def pstat(file):
        try:
            with open('persistence/'+file+'.p', 'rb') as f:
                data = pickle.load(f)
        except:
            data = np.zeros((len(strengths),len(thrs)))
            for s in range(len(strengths)):
                print(s)
                for n in range(ntot):
                    f = 'jun20rivers/%0.2f_0.15_90_tri_%d' % (strengths[s], n)
                    a = xy.DeltaNetwork.load(f)
                    for x in range(len(thrs)):
                        data[s,x] += stat(a, file, thr=thrs[x])
            data /= ntot

            with open('persistence/'+file+'.p', 'wb') as f:
                pickle.dump(data, f)

        return data

    data = pstat(f)

    plt.figure(figsize=(10,8))
    cmap = plt.get_cmap('coolwarm')
    for i in range(data.shape[0]):
        if strengths[i] in slabels:
            plt.plot(thrs, data[i,:], label=strengths[i],
                c=cmap(i/data.shape[0]))
        else:
            plt.plot(thrs, data[i,:], c=cmap(i/data.shape[0]))

    plt.xscale('log')
    plt.xlabel('Threshold conductivity')
    plt.ylabel(ylabels[f])
    plt.legend()
    plt.savefig('final/persistence/sim.png',transparent=True, dpi=200)
    #plt.savefig('final/persistence/sim.svg',transparent=True,)

def datapersistence():
    deltas = T.keys()
    plt.figure(figsize=(10,8))
    usa = plt.get_cmap('coolwarm')
    for d in deltas:
        print(d)
        a = xy.read_json(d)
        x = []
        thrs = np.logspace(-5, 1, 30)
        for thr in thrs:
            x.append(stat(a, 'wbridgesnew', thr=thr))
        plt.plot(thrs, x, c=usa(np.log10(T[d])/5 + 2/5))
        #plt.text(thrs[0], x[0], d)

    slabels = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for s in slabels:
        plt.plot(thrs[-1], x[-1], '-', c=usa(np.log10(s)/5 + 2/5),
            label=str(s))
    plt.legend()

    plt.xscale('log')
    plt.xlabel('Normalized threshold conductivity')
    plt.ylabel('Fraction of channel area in loops')
    plt.tight_layout()
    plt.savefig('final/persistence/data.png', dpi=200, transparent=True)
    plt.savefig('final/persistence/data.svg', transparent=True)

def comparepersistence(file='wbridges'):
    strengths = np.logspace(-2, 3, 21)
    thrs = np.logspace(-5, -2, 30)
    slabels = 2*np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ntot = 8
    def pstat(file='wbridges'):
        try:
            with open('persistence/'+file+'.p', 'rb') as f:
                data = pickle.load(f)
        except:
            data = np.zeros((len(strengths),len(thrs)))
            for s in range(len(strengths)):
                print(s)
                for n in range(ntot):
                    f = 'jun20rivers/%0.2f_0.15_90_tri_%d' % (strengths[s], n)
                    a = xy.DeltaNetwork.load(f)
                    for x in range(len(thrs)):
                        data[s,x] += stat(a, file, thr=thrs[x])
            data /= ntot

            with open('persistence/'+file+'.p', 'wb') as f:
                pickle.dump(data, f)

        return data

    data = pstat(file)

    plt.figure(figsize=(10,14))
    ax1 = plt.subplot(211)
    cmap = plt.get_cmap('coolwarm')
    for i in range(data.shape[0]):
        if strengths[i] in slabels:
            plt.plot(thrs, data[i,:], label=strengths[i],
                c=cmap(i/data.shape[0]))
        else:
            plt.plot(thrs, data[i,:], c=cmap(i/data.shape[0]))

    plt.xscale('log')
    plt.ylabel('Percent loop channel area (simulations)')

    deltas = T.keys()
    plt.subplot(212, sharex=ax1)
    usa = plt.get_cmap('coolwarm')
    for d in deltas:
        print(d)
        a = xy.read_json(d)
        x = []
        thrs = np.logspace(-5, 1, 30)
        for thr in thrs:
            x.append(stat(a, file, thr=thr))
        plt.plot(thrs, x, c=usa(np.log10(T[d])/5 + 2/5))
        #plt.text(thrs[0], x[0], d)

    slabels = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for s in slabels:
        plt.plot(thrs[-1], x[-1], '-', c=usa(np.log10(s)/5 + 2/5),
            label=str(s))
    plt.legend()

    plt.xscale('log')
    plt.xlabel('Normalized threshold conductivity')
    plt.ylabel('Percent loop channel area (data)')
    plt.tight_layout()
    plt.show()

def sticksdraw():
    for d in T.keys():
        a = xy.newjson(d)
        if isinstance(a, str):
            continue
        ratio = np.amax(a.LEAF.Vertex[:,0])/np.amax(a.LEAF.Vertex[:,1])
        if ratio > 1.2:#d in ['Barnstaple', 'Orinoco']:
            fig = plt.figure(figsize=(6*ratio, 18))
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharey=ax1)
            ax3 = plt.subplot(313, sharey=ax1)
        else:
            fig = plt.figure(figsize=(18,6/ratio))
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132, sharex=ax1)
            ax3 = plt.subplot(133, sharex=ax1)

        print(d)
        ax1.set_title(d)
        a.plot('loops', showbounds=True, thr=1e-4, ax=ax1, drawspecial=False)
        xadj = 0.05*np.amax(a.LEAF.Vertex[:,0])
        yadj = 0.05*np.amax(a.LEAF.Vertex[:,1])
        ax1.set_xlim([-1*xadj, np.amax(a.LEAF.Vertex[:,0]+xadj)])
        ax1.set_ylim([-1*yadj, np.amax(a.LEAF.Vertex[:,1]+yadj)])

        b = xy.read_json(d)
        b.plot('sticks', thr=1e-4, showbounds=False, ax=ax2, drawspecial=False)
        #b.drawloops(thr=1e-4, ax=ax2)
        ax2.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
        ax2.set_xlim([-1*xadj, np.amax(b.LEAF.Vertex[:,0]+xadj)])
        ax2.set_ylim([-1*yadj, np.amax(b.LEAF.Vertex[:,1]+yadj)])

        b.thin()
        #b.smooth()
        b.plot('sticks', showbounds=False, ax=ax3, drawspecial=True)
        b.drawloops(thr=1e-4, ax=ax3)
        ax3.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
        ax3.set_xlim([-1*xadj, np.amax(b.LEAF.Vertex[:,0]+xadj)])
        ax3.set_ylim([-1*yadj, np.amax(b.LEAF.Vertex[:,1]+yadj)])

        plt.tight_layout()
        plt.show()
        #plt.savefig('sticks/'+d+'.png', dpi=200)
        fig.clear()

def mstdiff():
    sim.realdataplot('mstdiff', 'line')
    #sim.marshes_overlay('mstdiff')
    sim.align('mstdiff', plot=False)
    plt.xlim([1e-2, 1e4])
    plt.legend()
    #plt.savefig('final/stats/mstdiff.svg',transparent=True,)
    plt.savefig('final/main/mstdiff.png', transparent=True, dpi=200)

def mstdiffl():
    sim.realdataplot('mstdiffl', 'line')
    sim.align('mstdiffl', plot=False)
    plt.xlim([1e-2, 1e4])
    #plt.savefig('final/stats/mstdiffl.svg',transparent=True,)
    plt.savefig('final/stats/mstdiffl.png',transparent=True, dpi=200)

def scalefree():
    strengths = np.logspace(-2, 3, 21)
    thrs = np.logspace(-5, -2, 30)
    slabels = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ntot = 35
    def pstat(file='wbridges'):
        try:
            with open('persistence/'+file+'.p', 'rb') as f:
                data = pickle.load(f)
        except:
            data = np.zeros((len(strengths),len(thrs)))
            for s in range(len(strengths)):
                print(s)
                for n in range(ntot):
                    f = 'jun20rivers/%0.2f_0.15_90_tri_%d' % (strengths[s], n)
                    a = xy.DeltaNetwork.load(f)
                    for x in range(len(thrs)):
                        data[s,x] += stat(a, file, thr=thrs[x])
            data /= ntot

            with open('persistence/'+file+'.p', 'wb') as f:
                pickle.dump(data, f)

        return data

    data = pstat('wbridges')

    plt.figure(figsize=(8,9))
    plt.plot(strengths, thrs[np.argmax(data < 0.2, axis=1)], 'b-')
    plt.xscale('log')
    plt.yscale('log')

    deltas = list(T.keys())
    #plt.figure(figsize=(8,6))
    thrs = np.logspace(-5, 1, 50)
    inds = []
    for d in deltas:
        print(d)
        a = xy.read_json(d)
        x = []
        for thr in thrs:
            x.append(stat(a, 'wbridges', thr=thr))
        inds.append(np.argmax(np.array(x)<0.2))

    Tstar = np.array(list(T.values()))/np.pi
    #plt.errorbar(Tstar, thrs[inds], c='k', ls='', ms=10,
    #    xerr=[0.5*Tstar, 2*Tstar], lw=0.2)
    plt.plot(Tstar, thrs[inds], 'g.', ms=14)
    for d in range(len(deltas)):
        if deltas[d] in ['Orinoco', 'Rajang']:
            plt.text(0.65*Tstar[d], 1.3*thrs[inds[d]],
                deltas[d], fontsize=14, alpha=0.5)
        elif deltas[d] in ['Barnstable']:
            plt.text(0.2*Tstar[d], 1.6*thrs[inds[d]],
                deltas[d], fontsize=14, alpha=0.5)
        elif deltas[d] in ['Ras Isa']:
            plt.text(0.5*Tstar[d], 1.3*thrs[inds[d]],
                deltas[d], fontsize=14, alpha=0.5)
        elif deltas[d] in ['Mississippi']:
            plt.text(0.65*Tstar[d], 1.3*thrs[inds[d]],
                deltas[d], fontsize=14, alpha=0.5)
        else:
            plt.text(0.65*Tstar[d], 1.02*thrs[inds[d]],
                deltas[d], fontsize=14, alpha=0.5)
    plt.xlabel(r'$T^*$')
    plt.ylabel(r'Threshold for at least 20\% channel area in loops')
    plt.savefig('final/stats/scalefree.png', transparent=True, dpi=150)
    #plt.savefig('final/main/scalefree.svg', transparent=True)

#nloops_data()
#loopareas_data()
sim.realdataplot('mstdiff')
