import time
import numpy as np
from matplotlib.collections import LineCollection
import scipy.optimize
from scipy.interpolate import griddata
import matplotlib.cm as cm
import matplotlib.pyplot as plt
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import networkx as nx

def plotter(fn):
    """
        A decorator that cleans the matplotlib figure given as
        the keyword argument 'figure' of the decorated class method,
        calls the drawing function,
        and then asks matplotlib to draw.
        If no kwarg 'figure' is provided, a new figure is maked.
        If the kwarg 'filename' is provided, plotting is done
        to the given file.
    """
    def wrapped(*args, **kwargs):
        if 'figure' in kwargs:
            plt.figure(num=kwargs['figure'])
            del kwargs['figure']
        else:
            plt.figure()

        if 'filename' in kwargs:
            filename = kwargs['filename']
            del kwargs['filename']
        else:
            filename = None

        plt.clf()
        plt.cla()

        ret = fn(*args, **kwargs)

        plt.title(args[0].param_str())

        if filename != None:
            plt.savefig(filename)
        else:
            plt.draw()
            plt.show()

        return ret

    return wrapped

class VascularNetwork(object):
    def __init__(self, LEAF, LSHP, size, C0=None, print_stats=False):
        """
            Constructs a new vascular network from the given
            Topology instance LEAF.
            A vascular network is defined as a collection of
            vertices and bonds between them, which can have a
            given conductivity.
            If no conductivites are provided, they are set to zero.
            This is essentially a wrapper for easier access to the
            parameters.

            size is supposed to be the network size parameter.
            C0 is a vector of initial conductivities.

            if print_stats is True, prints out basic network
            statistics such as number of bonds and nodes
        """
        super(VascularNetwork, self).__init__()

        # Geometric parameters
        self.LEAF = LEAF
        self.LSHP = LSHP

        self.bonds = LEAF.Bond.shape[0]
        self.verts = LEAF.Vertex.shape[0]

        # normalize all length scales
        self.bond_lens = self.bond_lengths()
        scale = self.bond_lens.max()

        self.bond_lens /= scale
        self.intersection_lens = self.LEAF.RidgeLengths/scale
        self.cell_areas = self.LEAF.CellAreas/scale**2

        # Initialize these variables so the can be used for caching
        self.bond_dists_sqr = None
        self.vert_dists_sqr = None

        # Computational buffers/results
        # self.Q = np.zeros((self.verts, self.verts))
        # self.q = np.zeros(self.verts)
        # self.CM = np.zeros((self.verts, self.verts))
        # self.p = np.zeros(self.verts)
        # self.G = np.zeros((self.verts, self.verts))
        # self.DpM = np.zeros((self.verts, self.verts))

        # save the bonds that neighbor the sinks
        '''self.ind_sink0 = [i for i in range(self.bonds) \
            if self.LEAF.Bond[i,0] == 0]
        self.ind_sink1 = [i for i in range(self.bonds) \
            if self.LEAF.Bond[i,1] == 0]'''

        self.size = size

        self.I_mat = self.incidence_matrix()
        self.I_mat_red = self.I_mat[:,1:]
        #self.triangles = self.triangle_list()
        #self.triang_mat = self.triangle_matrix()

        if C0 is not None:
            if len(C0) == self.bonds:
                self.C = C0
            else:
                raise Exception("Initial conductivities have wrong dimension!")
        else:
            self.C = -np.log(np.random.random(self.bonds))

        if print_stats:
            print ("---")
            print ("Initialized network.")
            print ("Vertices: {}".format(self.verts))
            print ("Bonds: {}".format(self.bonds))
            print ("---")

    def incidence_matrix(self):
        """ Construct a sparse oriented incidence matrix from
        the Bond list
        """
        nodes = list(self.LEAF.Bond[:,0]) + list(self.LEAF.Bond[:,1])
        edges = 2*np.array(range(self.bonds)).tolist()
        data = self.bonds*[1] + self.bonds*[-1]

        I_mat = scipy.sparse.coo_matrix((data, (edges, nodes)),
                shape=(self.bonds, self.verts))

        return I_mat.tocsc()

    def triangle_list(self):
        """
        Return a list of triangles in the network
        """
        triangles = set()
        blist = [list(x) for x in list(self.LEAF.Bond)]

        for u, v in blist:
            for w in range(self.verts):
                if sorted([u, w]) in blist and sorted([v, w]) in blist:
                    triangles.add(tuple(sorted([u, w, v])))

        return list(triangles)

    def triangle_matrix(self):
        """ Construct and return the triangle matrix which has
        one row for each triangle and verts columns.
        For each triangle, all entries the correspond to the
        triangle vertices are 1, the rest 0.
        """
        triangs = self.triangle_list()

        tri = np.repeat(range(len(triangs)), 3)
        nodes = np.array(triangs).flatten()

        data = ones(len(nodes))

        return scipy.sparse.coo_matrix((data, (tri, nodes)),
                shape=(len(triangs), self.verts)).tocsc()

    def C_matrix(self, C):
        """
            Constructs the conductivity matrix from the conductivity vector,
            or equivalently any M_{ij} from a list of values
            on the bonds.
        """
        # Make Cond into matrix
        CRes = scipy.sparse.coo_matrix((C, (self.LEAF.Bond[:,0],
            self.LEAF.Bond[:,1])), shape=(self.verts, self.verts))
        CRes = CRes.todense().copy()
        CRes = np.array(CRes + CRes.T)

        return CRes

    def C_matrix_sparse(self, C):
        """
            Constructs the conductivity matrix from the conductivity vector,
            or equivalently any M_{ij} from a list of values
            on the bonds.
        """
        # Make Cond into matrix
        CRes = scipy.sparse.coo_matrix((C, (self.LEAF.Bond[:,0],
            self.LEAF.Bond[:,1])), shape=(self.verts, self.verts))
        CRes = CRes + CRes.T;

        return CRes

    def C_matrix_asym(self, C):
        """
            Constructs the conductivity matrix from the conductivity vector,
            or equivalently any M_{ij} from a list of values
            on the bonds.
        """
        # Make Cond into matrix
        CRes = scipy.sparse.coo_matrix((C[:self.bonds],
            (self.LEAF.Bond[:,0], self.LEAF.Bond[:,1])), \
            shape=(self.verts, self.verts))

        CRes2 = scipy.sparse.coo_matrix((C[self.bonds:],
            (self.LEAF.Bond[:,1], self.LEAF.Bond[:,0])), \
            shape=(self.verts, self.verts))

        return np.array((CRes + CRes2).todense())

    def Q_matrix(self, CM, p):
        """
            Constructs the Q matrix from the C matrix and
            the p vector via fast np operations.

            We have Q_{ij} = C_{ij} (p_j - p_i).
        """

        DpM = np.repeat(p[np.newaxis, :], self.verts, axis=0)
        tmp = DpM - DpM.T
        DpM = tmp

        return CM*DpM

    def Q_matrix_asym(self, CM, p):
        """
            Constructs the Q matrix from the C matrix and
            the p vector via fast np operations.

            We have Q_{ij} = C_{ij} p_j - C_{ji} p_i.
        """
        #print (CM*p).flatten().nonzero()
        #print (CM*p[:,np.newaxis]).flatten().nonzero()
        CMT = CM.T.copy()
        Q = (2*CM + CMT)*p - (CM + 2*CMT)*p[:,np.newaxis]

        #Q = CM*p - CMT*p[:,np.newaxis]
        return Q

    def Q_vector(self, C, p):
        Q = C*(p[self.LEAF.Bond[:,0]] - p[self.LEAF.Bond[:,1]])
        return Q

    def Q_vector_asym(self, C, p):
        Q = C[:self.bonds]*p[self.LEAF.Bond[:,1]] - \
                C[self.bonds:]*p[self.LEAF.Bond[:,0]]
        return concatenate((Q, -Q))

    def G_matrix(self, CM):
        """
            Constructs the G matrix from the conductivity matrix
        """
        tvec = np.sum(CM, axis=1)

        return diag(tvec, 0) - CM

    def G_matrix_sparse(self, CM):
        """
            Constructs the G matrix from the conductivity matrix
        """
        tvec = CM.sum(axis=0)

        return (scipy.sparse.spdiags(tvec, [0], tvec.size, tvec.size,
            format='coo') - CM)

    def R_matrix_sparse(self, CM):
        """Returns resistances matrix, for resistance distance"""
        CM = CM.power(-1)
        tvec = CM.sum(axis=0)

        return (scipy.sparse.spdiags(tvec, [0], tvec.size, tvec.size,
            format='coo') - CM)

    def adjacency_matrix(self, thr=1e-10):
        """
            Returns the unweighted adjacency matrix.
            conductivities smaller than threshold are discarded
        """
        CM = self.C_matrix(self.C)

        return (CM > thr).astype(int)

    def adjacency_matrix_asym(self, thr=1e-10):
        """
            Returns the unweighted adjacency matrix for the directed graph
            defined by self.LEAF.Bond. A_vu = 1 if there is a bond (uv)
        """
        return scipy.sparse.coo_matrix((np.ones(self.LEAF.Bond.shape[0]),
            (self.LEAF.Bond[:,1], self.LEAF.Bond[:,0])),
            shape=(self.verts, self.verts))

    def laplacian_matrix(self):
        """
            Returns the unweighted Laplacian matrix.
            Uses adjacency_matrix
        """

        A = self.adjacency_matrix()

        return diag(np.sum(A, axis=0)) - A

    def effective_gamma(self):
        """
            Calculates the effective scaling exponent gamma
            by least squares fitting the scaling relation.
            The exponent is defined by

            C \sim Q^{2/(1+gamma)}.

            returns
            gamma: the (approximate) effective scaling exponent

            If the approximate scaling exponent cannot be calculated
            because there is no data (for whatever reason), returns
            zero.
        """
        Qs = log(self.Q.copy()**2).flatten()
        Cs = log(self.C_matrix(self.C)).flatten()

        keep = logical_and(logical_and(logical_and(isfinite(Qs),
            isfinite(Cs)), Cs > -10), Qs > -10)
        Qk = Qs[keep]
        Ck = Cs[keep]

        if Qk.shape[0] == 0:
            return 0

        A = vstack([Qk, ones(len(Qk))]).T

        x, res, rank, s = linalg.lstsq(A, Ck)

        #print "lstsq residuum:", res[0]
        #plt.figure()
        #plt.plot(Qk, Ck, 'o')
        #plt.show()
        #raw_input()

        return 1.0/x[0] - 1.0

    def topology_data(self):
        """
            Computes and returns

            number of topological vertices
            Strahler number
            bifurcation ratios
            ramification matrix

            We numerically check whether the network is a tree
            (actually, only if it has cycles). If it is not,
            None's are returned.
        """
        A = self.adjacency_matrix()

        if not is_acyclic(A):
            return None, None, None, None

        orders = np.zeros(self.verts)
        biorders = np.zeros((self.verts, self.verts))

        S = strahler_traversal(A, orders, biorders)

        topol_n = np.sum(orders)

        max_ord = max(where(orders != 0)[0])

        bif = orders[1:max_ord]/orders[2:(max_ord+1)]

        ram = biorders[:(max_ord + 1),:(max_ord + 1)]

        di = diag_indices(max_ord + 1)
        diag_entries = np.zeros(max_ord + 1)
        diag_entries[1:] = diag(ram)[:-1]
        ram[di] = diag_entries
        ram[1:,:] /= orders[1:max_ord+1,None]

        return topol_n, S, bif, ram[1:,:]

    def degree_distribution(self):
        """
            Calculates the degree distribution
            associated with the network graph.
        """
        A = self.adjacency_matrix()

        D = np.sum(A, axis=0)

        l = max(D)
        dist = np.zeros(l)

        # We ignore non-connected lattice points
        for i in xrange(1, l):
            dist[i] = (D == i).sum()

        return dist/np.sum(dist)

    def degrees(self, thr=1e-4):
        """ Return the vector of unweighted degrees of the network
        """
        # find threshold if network is a tree
        #thresh = sorted(self.C, reverse=True)[-self.verts+1]
        D = self.adjacency_matrix(thr=thr).sum(axis=0)

        return D

    def mean_degree(self):
        """ Return the mean degree of the network
        """
        # find threshold if network is a tree
        #thresh = sorted(self.C, reverse=True)[-self.verts+1]
        D = self.adjacency_matrix(threshold=1e-8).sum(axis=0)

        return D.mean()

    def std_degree(self):
        """ Return the standard deviation of the degree distribution
        """
        D = self.adjacency_matrix(threshold=1e-8).sum(axis=0)

        return D.std(ddof=1)

    def mean_weighted_degree(self):
        """ Return the mean weighted degree, where
        the weight is given by the conductivity.
        """
        CM = self.C_matrix(self.C)
        D = CM.sum(axis=0)

        return D.mean()

    def branch_lengths(self):
        """ Return the distribution of branch lengths,
        where we count the lengths between two nodes in the
        tree of degree > 2.
        """
        G = nx.from_np_matrix(self.adjacency_matrix(threshold=1e-8))

        branch_lengths = []
        current_branch = 1
        for (u, v) in nx.dfs_edges(G, source=0):
            if G.degree(v) == 2:
                current_branch += 1
            else:
                branch_lengths.append(current_branch)
                current_branch = 1

        return np.array(branch_lengths)

    def bond_neighbors(self):
        """ Returns a list that for each bond contains
        an np.array of indices of nodes next to that particular bond.
        """
        ns = []
        for b in self.LEAF.Bond:
            # Find left/right node neighbors
            b_l_ns = [c[0] for c in self.LEAF.Bond if not np.array_equal(b, c)\
                    and (c[1] == b[0] or c[1] == b[1])]
            b_n_ns = [c[1] for c in self.LEAF.Bond if not np.array_equal(b, c)\
                    and (c[0] == b[0] or c[0] == b[1])]

            ns.append(np.array(list(set(b_l_ns + b_n_ns))))

        return ns

    def bond_neighbor_indices(self):
        """ Returns a list that for each bond contains
        an np.array of indices of bonds next to that particular bond.
        """
        ns = []
        for b in self.LEAF.Bond:
            # Find left/right node neighbors
            b_ns = [i for i in xrange(self.bonds) if \
                len(set(b).intersection(self.LEAF.Bond[i])) == 1]

            ns.append(np.array(b_ns))

        return ns

    def bond_coords(self):
        """
            Returns lists of the bonds' coordinates for plotting
        """

        a = np.arange(self.bonds)

        xs = np.array([self.LEAF.Vertex[self.LEAF.Bond[a,0],0],
                       self.LEAF.Vertex[self.LEAF.Bond[a,1],0]]).T

        ys = np.array([self.LEAF.Vertex[self.LEAF.Bond[a,0],1],
                       self.LEAF.Vertex[self.LEAF.Bond[a,1],1]]).T

        return xs, ys

    def bond_distances_sqr(self):
        """
            Returns the matrix d_ij containing the Euclidean distances
            squared between the midpoints of bonds i and j.
        """

        if self.bond_dists_sqr == None:
            py = 0.5*(self.LEAF.Vertex[self.LEAF.Bond[:,0],1] + \
                self.LEAF.Vertex[self.LEAF.Bond[:,1],1])
            px = 0.5*(self.LEAF.Vertex[self.LEAF.Bond[:,0],0] + \
                self.LEAF.Vertex[self.LEAF.Bond[:,1],0])

            dx = np.repeat(px[np.newaxis, :], \
                self.bonds, axis=0) - \
                np.repeat(px[:, np.newaxis], self.bonds, axis=1)

            dy = np.repeat(py[np.newaxis, :], \
                self.bonds, axis=0) - \
                np.repeat(py[:, np.newaxis], self.bonds, axis=1)

            self.bond_dists_sqr = dx*dx + dy*dy

        return self.bond_dists_sqr

    def vert_distances_sqr(self, verts=None):
        """
            Returns the matrix d_ij containing the Euclidean distances
            squared between the vertices i and j.
        """
        if verts is None:
            px = self.LEAF.Vertex[:,0]
            py = self.LEAF.Vertex[:,1]
        else:
            px = self.LEAF.Vertex[verts,0]
            py = self.LEAF.Vertex[verts,1]

        dx = np.repeat(px[np.newaxis, :], \
            len(px), axis=0) - \
            np.repeat(px[:, np.newaxis], len(px), axis=1)

        dy = np.repeat(py[np.newaxis, :], \
            len(py), axis=0) - \
            np.repeat(py[:, np.newaxis], len(py), axis=1)

        if verts is None:
            self.vert_dists_sqr = dx*dx + dy*dy
            return self.vert_dists_sqr
        else:
            return dx*dx + dy*dy

    def bond_lengths(self, normalize=False):
        """
            Returns a vector containing the bond lengths.
            If normalize is True, will normalize to the smallest
            length.
        """
        ls = np.linalg.norm(self.LEAF.Vertex[self.LEAF.Bond[:,0],:] - \
            self.LEAF.Vertex[self.LEAF.Bond[:,1],:], axis=1)

        if normalize:
            ls = ls/min(ls[ls>0])

        return ls

    def scale_plt_figure(self, ax=None):
        """
            Sets the correct scales for the current
            matplotlib figure
        """
        if ax == None:
            ax = plt.axes()

        ax.set_aspect('equal')

        try:
            ax.set_ylim([min(self.LEAF.Vertex[:,1]) - 0.05, \
                max(self.LEAF.Vertex[:,1]) + 0.05])
            ax.set_xlim([min(self.LEAF.Vertex[:,0]) - 0.05, \
                max(self.LEAF.Vertex[:,0]) + 0.05])
        except:
            print ('error scaling axis')
            ax.set_ylim([-1.3, 1.3])
            ax.set_xlim([-1.3, 1.3])

    def plot_conductivities_raw(self, process=lambda x: (x/np.amax(x))**0.25, \
        magn=2, col=np.array([0,0,0]), ax=None, scale_axis=True,
        rescale=False, alpha=True):
        if ax == None:
            ax = plt.gca()

        xs, ys = self.bond_coords()
        conds = process(self.C.copy())

        if rescale:
            # rescale all coordinates to lie inside [0,1]^2
            xs -= xs.min()
            xs /= xs.max()

            ys -= ys.min()
            ys /= ys.max()

        colors = np.zeros((len(ys),4))
        colors[:,:3] = col*np.ones((len(ys),3))
        if alpha:
            colors[:,3] = conds
        else:
            colors[:,3] = 1

        segs = np.array([np.array([xs[bond], ys[bond]]).T for bond in range(len(ys))])
        '''#Thresholding to plot only lines with C > thr
        a = segs[np.where(self.C.copy()>thr)]
        acond = conds[np.where(self.C.copy()>thr)]
        acolors = colors[np.where(self.C.copy()>thr)]'''

        line_segments = LineCollection(segs,
                               linewidths=conds*magn,
                               colors=colors,
                               capstyle='round')
        ax.add_collection(line_segments)

        if scale_axis:
            self.scale_plt_figure(ax=ax)

    def plot_conductivities_red(self, process=lambda x: x, magn=2,
            ax=None, rescale=False):
        """ Plot conductivities as red lines orthogonal
        to the edges
        """
        col = np.array([1., 0., 0.])
        if ax == None:
            ax = plt.gca()

        xs, ys = self.bond_coords()
        conds = process(self.C.copy())

        xs = np.array(xs)
        ys = np.array(ys)
        if rescale:
            # rescale all coordinates to lie inside [0,1]^2
            xs -= xs.min()
            xs /= xs.max()

            ys -= ys.min()
            ys /= ys.max()


        for i in range(self.bonds):
            alpha = 1 if conds[i] > 1e-1 else conds[i]
            color = tuple(list((1. - conds[i])*col) + [alpha])

            # rotate by 90 degrees
            dx = xs[i][1] - xs[i][0]
            dy = ys[i][1] - ys[i][0]

            dxx = 0.5*(dx + dy)
            dyy = 0.5*(dy - dx)

            xx = [xs[i][0] + dxx, xs[i][1] - dxx]
            yy = [ys[i][0] + dyy, ys[i][1] - dyy]

            # make half as long
            xx = [3./4*xx[0] + 1./4*xx[1], 3./4*xx[1] + 1./4*xx[0]]
            yy = [3./4*yy[0] + 1./4*yy[1], 3./4*yy[1] + 1./4*yy[0]]

            ax.plot(xx, yy, linewidth=magn*conds[i], \
                color=color)

        self.scale_plt_figure(ax=ax)

    def plot_conductivities_asym_raw(self, process=lambda x: (x/amax(x))**0.25, \
        magn=2, col=np.array([1., 1., 1.])):
        """
            Plots the leaf network to the current Matplotlib figure.
            It won't call matplotlib's draw, you have to do that yourself!

            process is a function used to process the conductivity vector.
            magn sets a magnification factor for vein thickness
            col allows you to specify a color for the conductivities.

            This is a re-implementation of the old displayNetworkF.

            Usage example:

            plt.figure(5)
            netw.plot_conductivities(process=lambda x: (x/max(x))**(0.5))
            plt.title("Concentrations")
            plt.draw()

            Use the plot_conductivities method if you're lazy!
        """
        xs, ys = self.bond_coords()
        conds = process(self.C.copy())

        for i in range(self.bonds):
            alpha = 1 if conds[i] > 1e-1 else conds[i]
            color = tuple(list((1. - conds[i])*col) + [alpha])

            x = xs[i].copy()
            y = ys[i].copy()

            x[0] = 0.5*(x[0] + x[1])
            y[0] = 0.5*(y[0] + y[1])

            plt.plot(x, y,
                    linewidth=magn*conds[i], color=color)

        for i in range(self.bonds):
            alpha = 1 if conds[i+self.bonds] > 1e-1 else conds[i+self.bonds]
            color = tuple(list((1. - conds[i+self.bonds])*col) + [alpha])

            x = xs[i].copy()
            y = ys[i].copy()

            x[1] = 0.5*(x[0] + x[1])
            y[1] = 0.5*(y[0] + y[1])

            plt.plot(x, y,
                    linewidth=magn*conds[i+self.bonds], color=color)

        self.scale_plt_figure()

    def plot_node_topol_raw(self, qty, process=lambda x: x):
        """
            Plots qty as a function of node topological distance from the root
            node.
        """
        A = self.adjacency_matrix()
        ords = float('inf')*np.ones(self.verts)
        order_nodes(A, ords)

        n = int(max(ords[np.where(np.isfinite(ords))]))

        qty = process(qty)

        vals = [ qty[np.where(ords == i)].mean() for i in np.xrange(n + 1) ]
        stds = [ qty[np.where(ords == i)].std() for i in np.xrange(n + 1) ]

        plt.xlim(-0.1, n + 0.1)
        plt.errorbar(np.arange(n + 1), vals, yerr=stds)

    def plot_bond_topol_raw(self, qty, process=lambda x: x):
        """
            Plots qty as a function of bond topological
            distance from the root node
        """
        A = self.adjacency_matrix()
        ords = float('inf')*ones(self.bonds)
        order_bonds(A, ords, self.LEAF.Bond)

        n = int(max(ords[where(isfinite(ords))]))

        qty = process(qty)

        vals = [ qty[where(ords == i)].mean() for i in xrange(n + 1) ]
        stds = [ qty[where(ords == i)].std() for i in xrange(n + 1) ]

        plt.xlim(-0.1, n + 0.1)
        plt.errorbar(arange(n + 1), vals, yerr=stds)

    @plotter
    def plot_conductivities_topol(self, figure=None, title="""Conductivities vs\
 topological distance"""):
        self.plot_bond_topol_raw(self.C)
        plt.title(title)

    @plotter
    def plot_conductivities(self, figure=None, title="", \
        proc_pow=0.25, ticks=True):
        """
            Plots the vascular network to given matplotlib figure
            with some reasonable default values
        """
        self.plot_conductivities_raw(process=lambda x: (x/amax(x))**proc_pow, \
            magn=4)
        plt.title(title)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if not ticks:
            plt.xticks([])
            plt.yticks([])

    @plotter
    def plot_conductivities_line(self, figure=None, title="Conductivities",
        fit=False, area_power=0.5):
        """
            Plots the conductivities as a 1d plot (useful for pine needle).

            fit: True/False whether there should be attempted to fit the
                law A(1 - x/L)**B to the conductivities.
                will return (A, B, L)
            area_power: C**area_power is proportional to cross-sectional
                area
        """
        if fit:
            size = np.sum(self.C**area_power > 1e-6)
            claw = self.line_fit(area_power)

        self.path_plot_raw(self.C_matrix(self.C**area_power))

        plt.title("Area distributions along tree paths")

        if fit:
            x = linspace(0, size + 1, num=512)
            #plt.plot(x, claw[0]*(1 - x/claw[2])**claw[1], \
            #    label="Fitted a(1-x/l)^b", linewidth=3)
            plt.legend()

            return claw

    def plot_currents_raw(self, currents, \
        process=lambda x: x, thresh=1e-6, \
        head_width=0.025, linewidth_magn=1):
        """
            Plots the currents to the current figure, with maximum
            freedom in options.

            process: processes the current matrix again
            thresh: the threshold for currents
            head_width: the arrows' head width
            linewidth_magn: a magnification factor for the
                line widths
        """
        currents = process(currents)
        x = self.LEAF.Vertex.copy()

        # Draw arrows
        for i in range(self.bonds):
            if currents[self.LEAF.Bond[i,0], self.LEAF.Bond[i,1]] > thresh:
                plt.arrow(x[self.LEAF.Bond[i,1],0], x[self.LEAF.Bond[i,1],1], \
                (x[self.LEAF.Bond[i,0],0] - x[self.LEAF.Bond[i,1],0])/2, \
                (x[self.LEAF.Bond[i,0],1] - x[self.LEAF.Bond[i,1],1])/2, \
                linewidth=linewidth_magn*currents[self.LEAF.Bond[i,0], \
                self.LEAF.Bond[i,1]], head_width=head_width)
            elif currents[self.LEAF.Bond[i,0], self.LEAF.Bond[i,1]] < -thresh:
                plt.arrow(x[self.LEAF.Bond[i,0],0], x[self.LEAF.Bond[i,0],1], \
                (x[self.LEAF.Bond[i,1],0] - x[self.LEAF.Bond[i,0],0])/2, \
                (x[self.LEAF.Bond[i,1],1] - x[self.LEAF.Bond[i,0],1])/2, \
                linewidth=linewidth_magn*currents[self.LEAF.Bond[i,0], \
                self.LEAF.Bond[i,1]], head_width=head_width)

        self.scale_plt_figure()

    def plot_current_vector_raw(self, currents, head_width=0.025,
            linewidth_magn=1, thresh=1e-6):
        x = self.LEAF.Vertex.copy()
        currents = 2*currents/currents.max()
        # Draw arrows
        for i in range(self.bonds):
            if currents[i] > thresh:
                plt.arrow(x[self.LEAF.Bond[i,1],0], x[self.LEAF.Bond[i,1],1], \
                (x[self.LEAF.Bond[i,0],0] - x[self.LEAF.Bond[i,1],0])/2, \
                (x[self.LEAF.Bond[i,0],1] - x[self.LEAF.Bond[i,1],1])/2, \
                linewidth=linewidth_magn*currents[i], head_width=head_width)
            elif currents[i] < -thresh:
                plt.arrow(x[self.LEAF.Bond[i,0],0], x[self.LEAF.Bond[i,0],1], \
                (x[self.LEAF.Bond[i,1],0] - x[self.LEAF.Bond[i,0],0])/2, \
                (x[self.LEAF.Bond[i,1],1] - x[self.LEAF.Bond[i,0],1])/2, \
                linewidth=linewidth_magn*currents[i], head_width=head_width)

        self.scale_plt_figure()


    @plotter
    def plot_currents(self, title="Currents", magn=2, hw=0.025):
        """
            Plots the currents to the given figure,
            with some reasonable defaults.
        """
        self.plot_currents_raw(self.Q.copy(), \
            process=lambda x: x/amax(abs(x)), \
            linewidth_magn=magn, head_width=hw)
        plt.title(title)

    @plotter
    def plot_node_qty(self, v, include_zero=False):
        self.plot_node_qty_raw(*args, **kwargs)

    def plot_node_qty_raw(self, v, include_zero=False, ax=None,
            colorbar=True):
        if include_zero:
            data = zip(v, xrange(len(v)))
        else:
            data = zip(v, xrange(1, len(v)+1))

        if ax == None:
            ax = plt.gca()

        xs = []
        ys = []
        for d, i in data:
            xs.append(self.LEAF.Vertex[i,0])
            ys.append(self.LEAF.Vertex[i,1])

        sc = ax.scatter(xs, ys, c=v, s=70, zorder=10, cmap='np.summer')

        if colorbar:
            plt.colorbar(sc)

    def plot_node_qty_mesh_raw(self, v, ax=None, colorbar=True, cax=None,
            colorbar_label="", zorder=-15, vmax=None, cmap='np.summer',
            rescale=False):
        """ Plot the node qty using pcolormesh.
        We choose a zorder=-15 by default so the rasterization threshold
        can be chosen appropriately.
        """
        if ax == None:
            ax = plt.gca()

        data = izip(v, xrange(len(v)))

        xs = []
        ys = []

        if rescale:
            xmin_inside = self.LEAF.Vertex[:,0].min()
            xmax_inside = self.LEAF.Vertex[:,0].max()
            ymin_inside = self.LEAF.Vertex[:,1].min()
            ymax_inside = self.LEAF.Vertex[:,1].max()

            dx = xmax_inside - xmin_inside
            dy = ymax_inside - ymin_inside

            for d, i in data:
                xs.append((self.LEAF.Vertex[i,0] - xmin_inside)/dx)
                ys.append((self.LEAF.Vertex[i,1] - ymin_inside)/dy)

            xs.extend((self.LEAF.RestVertices[:,0] - xmin_inside)/dx)
            ys.extend((self.LEAF.RestVertices[:,1] - ymin_inside)/dy)
        else:
            for d, i in data:
                xs.append(self.LEAF.Vertex[i,0])
                ys.append(self.LEAF.Vertex[i,1])

            xs.extend(self.LEAF.RestVertices[:,0])
            ys.extend(self.LEAF.RestVertices[:,1])

        xs = np.array(xs)
        ys = np.array(ys)

        xmin = xs.min()
        xmax = xs.max()
        ymin = ys.min()
        ymax = ys.max()

        v = concatenate((v, -ones(self.LEAF.RestVertices.shape[0])))

        #if rescale:
        #    xs -= xmin_inside
        #    xs /= xmax_inside
        #    ys -= ymin_inside
        #    ys /= ymax_inside
        #print xs.max()
        #print xmax_inside

        # interpolate on grid
        X, Y = mgrid[xmin:xmax:500j, ymin:ymax:500j]
        C = griddata((xs, ys), v, (X, Y), method='nearest')

        # plot image
        cm.get_cmap(cmap).set_under('white')
        sc = ax.pcolormesh(X, Y, C, cmap=cmap, vmin=0, vmax=vmax,
                zorder=zorder)

        if colorbar:
            if cax == None:
                cb = plt.colorbar(sc, ax=ax, label=colorbar_label)
            else:
                cb = plt.colorbar(sc, cax=cax, label=colorbar_label)
            cb.ax.tick_params(axis='x', direction='in', labeltop='on')

        return sc

    def path_plot_raw(self, matrix, label=None):
        """
            Plots the values in matrix (depending on bonds)
            along the tree paths.
        """
        paths = []
        tree_paths(self.CM.copy(), paths)

        for p in paths:
            pvals = [matrix[p[i], p[i+1]] for i in xrange(len(p)-1)]
            if label != None:
                plt.plot(pvals, linewidth=3, label=label)
            else:
                plt.plot(pvals, linewidth=3)

    def path_plot_vert_raw(self, vector, label=None):
        """
            Plots the values in vector (depending on vertices)
            along the tree paths.
        """
        paths = []
        tree_paths(self.CM.copy(), paths)

        for p in paths:
            if label != None:
                plt.plot(vector[np.array(p)], linewidth=3, label=label)
            else:
                plt.plot(vector[np.array(p)], linewidth=3)


    def line_fit(self, area_power):
        """
            Fits the law area = A(1-z/L)^B to the line of conductivities
            where area = conductivity^area_power
        """
        size = np.sum(self.C**area_power > 1e-3)
        try:
            R = fit_circle_law(self.C**area_power, size)[0]
        except:
            R = None

        return R

    def plot_lattice_raw(self):
        """
            Plots the lattice points.
        """
        plt.plot(self.LEAF.Vertex[:,0], self.LEAF.Vertex[:,1], "*")
        self.scale_plt_figure()

    @plotter
    def plot_lattice(self, title="Lattice points"):
        """
            Plots the lattice points
        """
        self.plot_lattice_raw()
        plt.title(title)

    def param_str(self):
        """
            Returns a string containing the network's parameters
        """
        return "gam: {}".format(self.gamma)

    def __repr__(self):
        """
            Returns a string representation of the vascular network
        """
        return "\nSize parameter: " + str(self.size)
