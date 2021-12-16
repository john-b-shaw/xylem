#!/usr/bin/env python
# Math and linear algebra
import numpy as np
import scipy.sparse

# Base object
import network

# Setting up LEAF object
import LEAFclass as LFCLSS
import matplotlib.path as pltth
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

# Useful for statistics
from shapely.geometry import Polygon, Point, MultiPoint
import networkx as nx

# Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Polygon as pltPolygon

# General
import time
import warnings
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16
})

"""
    xylem.py

    Contains classes for the optimization of Xylem networks.

    Henrik Ronellenfitsch 2012
    Adam Konkol 2019
"""

class NetworkSuite(network.VascularNetwork):
    """
        An expansion of the network.VascularNetwork class with functionality
        for simulation and analysis
    """
    # Initialization and creation
    def __init__(self, LEAF, LSHP, size, C0=None,
            inputs='multi_sources', n_sources=3, sourceinds=None,
            outputs='line', n_sinks=None, sink_fraction=None, sinkinds=None,):
        """Initialization:
            LEAF: Topology object describing where nodes are in the xy plane
            LSHP: LeafShape object giving the border of the network
            size: corresponds to the density of points in the network. the exact
                  number is proportional to the square root of total nodes
            sigma: width of gaussian fluctuations, if simulating vasculature
            cst: corresponds to strength of gaussian fluctuations for
                 vascular systems, or it corresponds to the maximum fraction of
                 output flow that each basin takes in or puts out for a
                 geography simulation
            gamma: parameter related to scaling law of the system being
                   simulated
            C0: initial conditions, array of conductivities to begin with. If
                None, then random conductivities are used
            inputs: See fluctuation_ensemble for possible options
            outputs: ^
            n_sinks: number of sinks
            d: If n_sinks not explicitly specified, this is the fraction of
               nodes that will become sinks
        """

        # essential
        super(NetworkSuite, self).__init__(LEAF, LSHP, size, C0=C0)

        self.weights_mode = 'animal'

        poly = Polygon(self.LSHP.polyedge)
        self.perimeter = poly.length
        self.area = poly.area

        self.set_sources(inputs, n_sources, sourceinds)
        self.set_sinks(outputs, n_sinks, sink_fraction, sinkinds)

        self.char_length = np.sqrt(self.area/self.n_sinks)

    @staticmethod
    # FIXME: 'semicircle' shape
    def make_LSHP(shape):
        """ Return a LSHP object with given shape.
        Possible shapes are
        'hexagon', 'circle', 'square', 'triangle'.
        """
        # Initialization of the network
        LSHP = LFCLSS.LeafShape(shape, np.pi/3)

        # Form appropriate lattice boundaries
        if shape == 'hexagon':
            theta = np.arange(0.,7.) * np.pi/3.
            LSHP.polyedge = np.array([np.cos(theta), np.sin(theta)]).T
        elif shape == 'circle' or shape == 0:
            theta = np.linspace(0, 2*np.pi, 100)
            LSHP.polyedge = np.array([np.cos(theta), np.sin(theta)]).T
        elif shape == 'semicircle':
            pass
        elif shape == 'square':
            LSHP.polyedge = np.array([[-1,-1], [1, -1], [1, 1], [-1, 1],
                                      [-1, -1]])
        elif shape == 'triangle':
            LSHP.polyedge = np.array([[-1, -1], [1, -1], [0, 1], [-1,-1]])
        elif shape == 'invtriangle':
            LSHP.polyedge = np.array([[-1, -1], [1, -1], [0, 1], [-1,-1]])
            LSHP.polyedge[:,1] *= -1
        elif shape == 'strip':
            x = 0.5
            LSHP.polyedge = np.array([[-x,-1], [x, -1], [x, 1], [-x, 1],
                                      [-x, -1]])
        elif shape == 'wide':
            x = 2
            LSHP.polyedge = np.array([[-x,-1], [x, -1], [x, 1], [-x, 1],
                                      [-x, -1]])
        elif shape == 'sine':
            amplitude = 0.3
            w = 0.5
            x = amplitude*np.sin(3*np.pi*np.linspace(0,1,20))
            y = np.linspace(-1,1,20)

            LSHP.polyedge = np.zeros((41,2))
            LSHP.polyedge[:20,0] = x + w
            LSHP.polyedge[:20,1] = y[::-1]
            LSHP.polyedge[20:40,0] = x - w
            LSHP.polyedge[20:40,1] = y
            LSHP.polyedge[40,:] = [x[0]+w, y[-1]]

        elif isinstance(shape, float):
            theta = np.linspace(0, 2*np.pi, 100)
            b = np.sqrt(1 - shape**2)
            LSHP.polyedge = np.array([b*np.cos(theta), np.sin(theta)]).T
        else:
            assert TypeError, 'Type %s is not supported' % shape

        return LSHP

    @staticmethod
    def make_LEAF(leafName, density, lattice, LSHP, yplot=False, angle=np.pi/3.,
                  noise=0.0, zoom_factor=1.0, shapeplot=False, stats=False,
                  trimming_percentile=100, ):
        """ Create a new network with given name, size, lattice type etc...
        noise is added as a percentage of np.mean bond length
        to the triangular and square lattices
        Parameters:
            leafName: LEAF class name
            density: Proportional to square root of number of nodes
            lattice: how nodes are distributed in the plane
            LSHP: Leaf shape (Topology class) instance
            yplot: option to plot Voronoi tesselation
            angle: only used for lattice = 'triangle'
            noise: determines how much nodes will np.differ from the lattice
            zoom_factor: ??
            shapeplot: option to plot leaf shape in its own plot
            stats: prints initial number of nodes and bonds
            trimming_percentile: bond length np.percentile above which the bonds are
                removed. Critical for square leafshape so there are not long
                edges along the leaf shape boundary.
        """
        def verts_plt_path(vertex):
            """ Returns a matplotlib Path object describing the polygon defined
            by vertices.
            """
            # Set up polygon
            verts = np.zeros((vertex.shape[0] + 1, vertex.shape[1]))
            verts[:-1,:] = vertex
            #verts[-1,:] = cycle.coords[0,:]

            codes = pltth.Path.LINETO*np.ones(verts.shape[0])
            codes[0] = pltth.Path.MOVETO
            codes[-1] = pltth.Path.CLOSEPOLY

            return pltth.Path(verts, codes)

        def replace_nan_by_avg(ar):
            # replace all nans by the average of the rest
            avg = ar[np.isfinite(ar)].mean()
            ar[np.isnan(ar)] = avg

            return ar

        def polygon_area(coords):
            """ Return the area of a closed polygon
            """
            Xs = coords[:,0]
            Ys = coords[:,1]

            # Ignore orientation
            return 0.5*abs(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))

        LEAF = LFCLSS.Topology(leafName, lattice)

        if lattice == 'yjunc':
            Mnei = 2
            X = np.linspace(-1.5, 1.5, num=density)
            Y = np.zeros(density)

            LEAF.height = X[1] - X[0]

            Y2 = np.arange(LEAF.height, 1, LEAF.height)
            X2 = X[int(len(X)/3)]*np.ones(len(Y2))

            maxlength = LEAF.height*1.01
            VertexM = np.zeros((density + len(Y2), 2))
            VertexM[:, 0] = np.concatenate((X, X2))
            VertexM[:, 1] = np.concatenate((Y, Y2))

        elif lattice == 'xjunc':
            Mnei = 2
            X = np.linspace(-1.5, 1.5, num=density)
            Y = np.zeros(density)

            LEAF.height = X[1] - X[0]

            Y2 = np.arange(LEAF.height, 1, LEAF.height)
            X2 = X[len(X)/3]*np.ones(len(Y2))

            Y3 = np.arange(-LEAF.height, -1, -LEAF.height)
            X3 = X[len(X)/3]*np.ones(len(Y3))

            maxlength = LEAF.height*1.01
            VertexM = np.zeros((density + len(Y2) + len(Y3), 2))
            VertexM[:, 0] = np.concatenate((X, X2, X3))
            VertexM[:, 1] = np.concatenate((Y, Y2, Y3))

        elif lattice == 'hjunc':
            X = np.linspace(-1.5, 1.5, num=density)
            Y = np.zeros(density)

            LEAF.height = X[1] - X[0]

            Y2 = np.arange(LEAF.height, 1, LEAF.height)
            X2 = X[len(X)/3]*np.ones(len(Y2))

            Y3 = np.arange(-LEAF.height, -1, -LEAF.height)
            X3 = X[len(X)/3]*np.ones(len(Y3))

            Y4 = np.arange(LEAF.height, 1, LEAF.height)
            X4 = X[len(X)/3 + 4]*np.ones(len(Y4))

            Y5 = np.arange(-LEAF.height, -1, -LEAF.height)
            X5 = X[len(X)/3 + 4]*np.ones(len(Y5))

            maxlength = LEAF.height*1.01
            VertexM = np.zeros((density + len(Y2) + len(Y3) + len(Y4) + len(Y5), 2))
            VertexM[:, 0] = np.concatenate((X, X2, X3, X4, X5))
            VertexM[:, 1] = np.concatenate((Y, Y2, Y3, Y4, Y5))

        # Generate Lattice
        elif lattice == 'random':
            """ We generate a lattice from Delaunay triangulation
            of random points on the plane
            """
            n_points = int(0.5*density**2)
            VertexM = np.random.random((n_points, 2))*2 + np.array([-1,-1])

            LEAF.height = max(VertexM[:,1]) - min(VertexM[:,1])

            maxlength = None

        elif lattice == 'triangle':
            x, y = np.meshgrid(np.linspace(-1,1,int(np.sqrt(density))),
                np.linspace(-1,1,int(np.sqrt(density))))
            x[::2, :] += (x[0,1] - x[0,0])/2

            if noise > 0.0:
                # move positions around randomly
                x += noise*3.2/density*(2*np.random.random(x.shape) - 1)
                y += noise*3.2/density*(2*np.random.random(y.shape) - 1)

            VertexM[:,0] = x.flatten()
            VertexM[:,1] = y.flatten()

        elif lattice == 'line':
            X = np.linspace(-1.5, 1.5, num=density)
            Y = np.zeros(density)

            LEAF.height = X[1] - X[0]

            maxlength = LEAF.height*1.01
            VertexM = np.zeros((density, 2))
            VertexM[:, 0] = X
            VertexM[:, 1] = Y

        elif lattice == 'square':
            x = np.linspace(-1, 1, density)
            y = np.linspace(-1, 1, density)

            maxlength = (x[1] - x[0])*(1.01 + 2*noise)

            x, y = [a.flatten() for a in np.meshgrid(x,y)]

            if noise > 0.0:
                # move positions around randomly
                x += noise*3.2/density*(2*np.random.random(x.shape) - 1)
                y += noise*3.2/density*(2*np.random.random(y.shape) - 1)

            VertexM = np.array([x, y]).T

        elif lattice == 'rect':
                x = np.linspace(0, 2.5, density)
                y = np.linspace(-1.05, 1.05, 2*density)

                maxlength = (x[1] - x[0])*1.01

                X, Y = np.meshgrid(x, y)

                X = np.reshape(X, (2*density**2, 1))
                Y = np.reshape(Y, (2*density**2, 1))

                x = X[:,0]
                y = Y[:,0]

                VertexM = np.array([x, y]).T

        else:
            # load lattice from text file
            VertexM = np.loadtxt(lattice, delimiter=',')
            n_points = VertexM.shape[0]

            VertexM *= 2.42
            VertexM += np.array([1.2, 0])

            LEAF.height = max(VertexM[:,1]) - min(VertexM[:,1])

            VertexM *= zoom_factor

            maxlength = None

        #VertexM[:,0] -= min(VertexM[:,0]);
        #VertexM[:,1] -= np.mean(VertexM[:,1]);

        xyleaf = LSHP.polyedge.T

        # change com of leafshape to mid node of
        # network if leafshape is a circle.
        '''if LSHP.comment == 'circle' or LSHP.comment == 'hexagon':
            com = VertexM.mean(axis=0)
            central_i = np.argmin(np.linalg.norm(VertexM - com, axis=1))
            central = VertexM[central_i]

            lshape = xyleaf.T
            lshape -= lshape[:-1,:].mean(axis=0)
            lshape += central

            xyleaf = lshape.T'''

        # remove vertices that are outside of the shape

        # Voronoi tesselation gives bonds directly
        vor = Voronoi(VertexM)
        BondM = vor.ridge_points.copy()

        # nxutils are deprecated
        path = verts_plt_path(xyleaf.T)
        Kall = path.contains_points(VertexM)

        orig_indices = np.where(Kall)[0]
        RestVertices = np.where(np.logical_not(Kall))[0]
        VertexM = VertexM[Kall,:]

        # remove all bonds that connect to removed vertices
        BondM = BondM[Kall[BondM[:,0]] & Kall[BondM[:,1]], :]

        # update indices

        # this map is the inverse of orig_indices
        #index_map = -np.ones(Kall.shape)
        #index_map[orig_indices] = arange(orig_indices.shape[0])

        # equivalent to the above but shorter
        new_indices = np.cumsum(Kall) - 1

        BondM[:,0] = new_indices[BondM[:,0]]
        BondM[:,1] = new_indices[BondM[:,1]]

        #remove outer higher length
        vecX = np.zeros(BondM.shape)
        vecX[:,0] = VertexM[BondM[:,0],0]
        vecX[:,1] = VertexM[BondM[:,1],0]
        vecY = np.zeros(BondM.shape)
        vecY[:,0] = VertexM[BondM[:,0],1]
        vecY[:,1] = VertexM[BondM[:,1],1]

        lens = np.sqrt(np.diff(vecX)**2 + np.diff(vecY)**2)
        if maxlength == None:
            maxlength = np.percentile(lens, trimming_percentile)

        K = (lens <= maxlength);

        BondM = BondM[np.squeeze(K),:].copy();

        Np2 = VertexM.shape[0]; #actual number of nodes
        LBondsM = BondM.shape[0]
        if stats:
            print('Number of initial nodes: %d' % Np2)
            print('Number of initial bonds: %d' % LBondsM)
        #BondM = sort(BondM, axis=1)

        #construct neighbor list
        # We never need this and it's buggy!
        #NeighM = None#neighborsF(BondM,Mnei,Np2)

        #figure out which bond belongs to which Voronoi ridge
        ridge_lens = np.zeros(BondM.shape[0])
        for i, (u, v) in enumerate(BondM):
            u, v = orig_indices[u], orig_indices[v]

            ridge_inds = np.where(np.all(vor.ridge_points == [u, v], axis=1))[0]

            if ridge_inds.size == 0:
                ridge_lens[i] = np.nan
                print( "Error: triangulation bond not in original voronoi tesselation")
                continue

            ridge_ind = ridge_inds[0]

            # find length of Voronoi ridge
            ridge_verts = vor.ridge_vertices[ridge_ind]

            if -1 in ridge_verts:
                # one is infinity, length is undefined
                ridge_lens[i] = np.nan
            else:
                ridge_lens[i] = np.linalg.norm(vor.vertices[ridge_verts[0]]
                        - vor.vertices[ridge_verts[1]])

        ridge_lens = replace_nan_by_avg(ridge_lens)

        # figure out what the area of each Voronoi cell is
        cell_areas = np.zeros(VertexM.shape[0])
        for i in range(VertexM.shape[0]):
            region = vor.point_region[orig_indices[i]]
            region_verts = vor.regions[region]

            if -1 in region_verts:
                cell_areas[i] = np.nan
            else:
                cell_verts = vor.vertices[region_verts]
                # make polygon closed
                cell_verts = np.vstack((cell_verts, cell_verts[0,:]))
                cell_areas[i] = polygon_area(cell_verts)

        cell_areas = replace_nan_by_avg(cell_areas)

        # find leftmost vertex and make it the zeroth one
        tempm = min(VertexM[:,0]);
        imin = np.argmin(VertexM[:,0]);
        Kleft = np.nonzero(abs(tempm -VertexM[:,0]) < 1e-6)
        isortleft = np.argsort(VertexM[Kleft[0],1]);

        mid_elem = isortleft[int(len(isortleft)/2.0)]
        imin = [Kleft[0][mid_elem]]

        # swap vertices
        VertexM[imin,:], VertexM[0, :] = VertexM[0,:], VertexM[imin,:]
        cell_areas[imin], cell_areas[0] = cell_areas[0], cell_areas[imin]

        # swap Bonds
        zero_entries = (BondM == 0)
        min_entries = (BondM == imin[0])

        BondM[zero_entries] = imin[0]
        BondM[min_entries] = 0

        # sort Bonds
        BondM.sort(axis=1)

        #set structure
        LEAF.Vertex = VertexM
        LEAF.RestVertices = RestVertices
        LEAF.Bond = BondM
        LEAF.RidgeLengths = ridge_lens
        LEAF.CellAreas = cell_areas
        LEAF.Voronoi = vor

        #plot (optional)
        if yplot:
            voronoi_plot_2d(vor)
            plt.plot(xyleaf[0,:], xyleaf[1,:])

            plt.show()

        return LEAF

    # simulation
    def flow_flux_en_weights(self):
        """ Return the weights for flows, fluxes, and energies
        """
        if self.weights_mode == 'plant':
            self.flow_wts = self.intersection_lens/self.bond_lens
            self.flux_wts = 1./self.bond_lens
            self.en_wts = self.intersection_lens/self.bond_lens
            self.cost_wts = self.intersection_lens*self.bond_lens
        elif self.weights_mode == 'animal':
            self.flow_wts = 1./self.bond_lens
            self.flux_wts = 1./self.bond_lens
            self.en_wts = 1./self.bond_lens
            self.cost_wts = self.bond_lens
        elif self.weights_mode == 'none':
            # Use this if C0 is specified and already includes the lengths
            self.flow_wts = 1.0
            self.flux_wts = 1.0
            self.en_wts = 1.0
            self.cost_wts = 1.0
        else:
            print ('Warning: neither plant nor animal, using unit weights')
            self.flow_wts = 1.0
            self.flux_wts = 1.0
            self.en_wts = 1.0
            self.cost_wts = 1.0

    def Q2_avg_vector(self, C, ds):
        CM = self.C_matrix_sparse(C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

        if ds.shape[1] > 1:
            Qsqr = np.sum((C[:,np.newaxis]*self.I_mat_red.dot(
                scipy.sparse.linalg.spsolve(G, ds[1:,:])))**2, axis=1)
        else:
            Qsqr = (C*self.I_mat_red.dot(
                scipy.sparse.linalg.spsolve(G, ds[1:,:])))**2

        Qsqr /= ds.shape[1]

        return Qsqr

    def set_sources(self, inputs, n_sources, sourceinds):
        self.sources = []
        self.n_sources = n_sources

        self.inputs = inputs
        if self.inputs == 'none':
            self.n_sources = 0
            self.sources = []
            return self.sources
        if sourceinds is not None:
            self.sources = sourceinds
            self.n_sources = len(self.sources)
            return self.sources

        if self.inputs == 'center':
            A = np.array([[0,0]])
            self.n_sources = 1
        elif self.inputs == 'upper_corners':
            x = [-1, 1]
            y = [1, 1]
            A = np.array([x,y]).T
        elif self.inputs == 'multi_sources': #sources around circle perimeter
            x = [np.cos(a*2*np.pi/n_sources) for a in
                 np.arange(0,self.n_sources)]
            y = [np.sin(a*2*np.pi/n_sources) for a in
                 np.arange(0,self.n_sources)]
            A = np.array([x,y]).T
        elif self.inputs == 'line_source':
            deltax = 2/(n_sources+1)
            x = [deltax*(i-(n_sources-1)/2) for i in
                 range(self.n_sources)]
            y = np.ones_like(x)*max(self.LEAF.Vertex[:,1])
            A = np.array([x,y]).T
        elif self.inputs == 'line_source_wide':
            x = [2*(1+i)/(n_sources+1)-1 for i in range(n_sources)]
            y = np.ones_like(x)*max(self.LEAF.Vertex[:,1])
            A = np.array([x,y]).T
        else:
            raise ValueError('Given source term not implemented')

        for i in range(self.n_sources):
            distsq = (self.LEAF.Vertex[:,0] - A[i,0])**2 + \
                     (self.LEAF.Vertex[:,1] - A[i,1])**2
            self.sources.append(np.argmin(distsq))

    def set_sinks(self, outputs, n_sinks, sink_fraction, sinkinds):
        self.outputs = outputs
        if sinkinds is not None:
            self.sinks = sinkinds
            self.n_sinks = len(self.sinks)
            return self.sinks

        self.n_sinks = n_sinks
        if sink_fraction != None:
            self.n_sinks = int(sink_fraction*self.verts)

        self.sinks = []
        if self.outputs == 'random':
            """Randomly distributed sinks with Gaussian fluctuations"""

            p = np.ones(self.verts)
            p[self.sources] = 0
            p /= self.verts - self.n_sources #Uniform probability for sinks
            self.sinks = np.random.choice(self.verts, size=self.n_sinks, p=p,
                                          replace=False)
            return self.sinks
        if self.outputs == 'grid':
            """Sinks chosen closest to a grid"""

            r = self.char_length
            x = np.arange(-1,1,r)
            #y = np.arange(-1,1,r)
            X_all = np.zeros((len(x)**2,2))
            x0 = []
            y0 = []
            #this double for loop can be much more efficient using np.meshgrid
            for i in range(len(x)):
                for j in range(len(x)):
                    x0.append(x[i])
                    y0.append(x[j])
            X_all[:,0] = np.array(x0)
            X_all[:,1] = np.array(y0)
            A = np.array([X_all[i,:] for i in range(len(x0)) if
                             (X_all[i,0]**2+X_all[i,1]**2)<(1.40)**2])
            A[:,0] += 1.45
            self.sinks = []
        elif self.outputs == 'line':
            x = [2*(1+i)/(self.n_sinks+1)-1 for i in range(self.n_sinks)]
            y = np.ones_like(x)*min(self.LEAF.Vertex[:,1])
            A = np.array([x,y]).T
            self.sinks = []
        elif self.outputs == 'semicircle':
            assert self.LSHP.comment == 'circle' or self.LSHP.comment == \
                'ellipse', 'semicircle requires circular leaf shape (LSHP)'
            x = [np.sin((np.pi/self.n_sinks)*(i-(self.n_sinks-1)/2))
                 for i in range(self.n_sinks)]
            y = [-np.cos((np.pi/self.n_sinks)*(i-(self.n_sinks-1)/2))
                 for i in range(self.n_sinks)]
            A = np.array([x,y]).T
            self.sinks = []
        elif self.outputs == 'circle':
            assert self.LSHP.comment == 'circle' or \
                isinstance(self.LSHP.comment, float), \
                'semicircle requires circular leaf shape (LSHP)'
            x = [np.sqrt(1-self.LSHP.comment**2) * \
                 np.sin((2*np.pi/self.n_sinks) * \
                 (i-(self.n_sinks-1))) for i in range(self.n_sinks)]
            y = [np.cos((2*np.pi/self.n_sinks)*(i-(self.n_sinks-1)))
                 for i in range(self.n_sinks)]
            A = np.array([x,y]).T
            self.sinks = []
        elif self.outputs == 'outer_spaced':
            s = self.perimeter / n_sinks
            A = np.zeros((n_sinks, 2))
            A[0,:] = self.LSHP.polyedge[0,:]
            n = 1 #next node to check
            for i in range(1, n_sinks):
                if np.linalg.norm(self.LSHP.polyedge[n,:] - A[i-1,:]) < s:
                    d = np.linalg.norm(self.LSHP.polyedge[n,:] - A[i-1,:])
                    n += 1
                    while d + np.linalg.norm(self.LSHP.polyedge[n,:] - \
                        self.LSHP.polyedge[n-1,:]) < s:
                        d += np.linalg.norm(self.LSHP.polyedge[n,:] - \
                            self.LSHP.polyedge[n-1,:])
                        n += 1
                    t = self.LSHP.polyedge[n,:] - self.LSHP.polyedge[n-1,:]
                    A[i,:] = self.LSHP.polyedge[n-1,:] + (s-d) * \
                        (t/np.linalg.norm(t))
                else:
                    t = self.LSHP.polyedge[n,:] - self.LSHP.polyedge[n-1,:]
                    A[i,:] = A[i-1,:] + (t/np.linalg.norm(t))
        elif self.outputs == 'invtriangle':
            x = np.linspace(-1, 1, self.n_sinks)
            y = np.abs(-2*x) - 0.95
            A = np.array([x,y]).T
            self.sinks = []

        poly = Polygon(self.LSHP.polyedge)
        A = A[[poly.contains(Point(A[n,:])) for n in range(A.shape[0])],:]

        for i in range(A.shape[0]):
            distsq = (self.LEAF.Vertex[:,0] - A[i,0])**2 + \
                      (self.LEAF.Vertex[:,1] - A[i,1])**2
            sink = np.argmin(distsq)
            if sink not in self.sources and sink not in self.sinks:
                self.sinks.append(sink)
                self.LEAF.Vertex[sink,:] = A[i,:]

        self.n_sinks = len(self.sinks)

    def simulate_base(self, dt=0.1, timesteps=1e10, converge_en=1e-10,
            plot=False, plot_interval=10, movie_dir='movie', entropy=False):
        """ Simulates the xylem network as a dynamical system
        by integrating dC/dt = f(Q) - C + driving

        Parameters:
            dt: Integration time step
            timesteps: number of time steps to integrate
            converge_en: stopnp.ping criterion for relative
                change in energy
            kappa: nondimensional background production
            plot: whether to plot the network dynamics
        """

        if plot:
            fig, ax = plt.subplots(1)

        self.ds = self.fluctuation_ensemble()
        self.flow_flux_en_weights()
        q_power = 1./(self.gamma + 1)

        en_last = 0
        k = 0 #for plotting
        for i in range(1, int(timesteps)):
            # Calculate average of squared currents for BCs given in self.ds
            Q2 = self.Q2_avg_vector(self.C*self.flux_wts, self.ds)

            if entropy:
                if i == 1:
                    self.entropy = np.array([self.graph_entropy()])
                else:
                    self.entropy = np.append(self.entropy,
                        [self.graph_entropy()], axis=0)

            # Adaptation equation
            self.C += dt*(Q2**q_power - self.C)

            # dissipation
            nonz = self.C > 1e-8
            en = sum(Q2[nonz]/(self.en_wts[nonz]*self.C[nonz]))

            if plot and i % plot_interval == 0:
                print('Frame: %d' % k)

                # make video
                fig.clear()
                ax = fig.add_subplot(111)

                plt.axis('off')
                self.plot(drawspecial=False)
                #self.plot(showscale=True, drawspecial=False)
                '''ax.set_title('$t=%0.1f, \sigma=%0.1f, c=%0.1f$' % (i*dt,
                             self.sigma_ratio, self.cst), fontsize=14)'''
                ax.set_title('$t=%0.1f$' % (i*dt), fontsize=18)
                fig.savefig(movie_dir + '/%05d.png' % k, dpi=300)

                self.drawspecial()
                fig.savefig(movie_dir + '/a%05d.png' % k, dpi=300)

                k += 1

                if i > 1000:
                    plot = False

            # break when energy converges
            if i == 0:
                pass
            else:
                if en_last == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        delta_en = abs((en - en_last)/en_last)
                else:
                    delta_en = abs((en - en_last)/en_last)
                if delta_en < converge_en:
                    print("Converged")
                    break

            en_last = en

            if np.any(np.isnan(self.C)):
                print("Error in simulation \nEnding simulation")
                break # something is wrong and we need to stop.

        self.energy_value = en
        #self.entropy_value = mean(self.flow_entropies(C))
        #self.edge_entropy_value = mean(self.edge_entropies(C))
        #self.cycle_number = self.cycles(C)
        #self.cut_bond_energy_value = self.cut_bond_energy(C)
        #self.normal_energy_value = self.energy(C)

        self.time_steps = i
        print('Time steps: ', self.time_steps)

    # loop statistics
    def find_cycles_backup(self):
        """Finds something close to the minimal cycle basis of the network
        Avoid using
        Note: This is not necessarily the most efficient implementation.
            The method that Claire Dore wrote which prunes all trees from the
            graph and finds loops by always taking the path of the leftmost
            edge can be around 10x faster. However, using nx.cycle_basis
            into nx.minimum_cycle_basis:
                1) is easier to use
                2) requires substantially less code and fewer libraries
                3) is fairly intuitive as a change of basis
        Returns:
            self.cs: a list containing the lists of nodes in each cycle
        """
        if hasattr(self, 'cycles'):
            return self.cycles
        else:
            self.to_networkx()

            #Find indices of nodes in the cycle basis, not necessarily minimal
            cycles = nx.cycle_basis(self.G)
            keep = np.unique([node for cycle in cycles for node in cycle])

            #Get rid of every non-cycle node
            G2 = self.G.copy()
            G2.remove_nodes_from(np.delete(np.arange(self.verts), keep))

            #Find the minimum cycle basis
            cycles = nx.minimum_cycle_basis(G2)

            #Order the nodes in each cycle
            for i in range(len(cycles)):
                new_cycle = [cycles[i][0]]

                searching = True
                while searching:
                    neighbors = self.G.neighbors(new_cycle[-1])
                    for neighbor in neighbors:
                        if neighbor in cycles[i] and neighbor not in new_cycle:
                            new_cycle.append(neighbor)
                            break

                    if len(new_cycle) == len(cycles[i]):
                        searching = False

                cycles[i] = new_cycle

            self.cycles = cycles
            return self.cycles

    def find_cycles(self, thr=1e-6):
        """ A fast algorithm for finding the regions of a spatial network,
            where a region is an area enclosed by a loop that contains no
            other nodes of edges. The algorithm retains all nodes in a cycle
            basis of G, removes isolated loops (where all nodes have degree 2),
            splits the graph into connected component subgraphs, then finds all
            loops in each subgraph. Loops containing regions are found using a
            leftmost edge search starting from nodes of degree > 2. Nodes of
            degree > 2 on the outside of the subgraph will yield the entire
            exterior of the subgraph by a leftmost edge search when going
            clockwise, so this is done first to avoid clockwise searching when
            starting from a node on the exterior of the subgraph.
        Returns:
            self.cycles: a list of lists of nodes in each cycle (ordered, not
                oriented either cw or ccw)
        """
        if hasattr(self, 'cycles'):
            return self.cycles
        else:
            def find_left_node(G, start, end):
                neighbors = list(G.neighbors(end))
                neighbors.remove(start)

                if len(neighbors) == 1:
                    return neighbors[0]
                else:
                    orig = (self.LEAF.Vertex[end,:] - \
                            self.LEAF.Vertex[start,:]).dot(np.array([1,1j]))

                    neighs = (self.LEAF.Vertex[neighbors,:] - \
                              self.LEAF.Vertex[end,:]).dot(np.array([1,1j]))

                    angles = np.angle(neighs/orig)
                    return neighbors[np.argmax(angles)]

            def find_loop(G, start, end):
                loop = [start, end]
                while loop[-1] != loop[0]:
                    loop.append(find_left_node(G, loop[-2], loop[-1]))
                return loop

            def find_exterior(sg):
                hull = ConvexHull(self.LEAF.Vertex[sg.nodes(),:])
                hullverts = np.array(sg.nodes())[hull.vertices]
                hullstarts = [n for n in hullverts if sg.degree(n) == 2]

                if len(hullstarts) == 0:
                    raise IndexError('loop has no exterior nodes of deg 2')

                neighbors = list(sg.neighbors(hullstarts[0]))
                #cross product negative if sequence is going clockwise
                a = self.LEAF.Vertex[hullstarts[0],:] - \
                    self.LEAF.Vertex[neighbors[0],:]
                b = self.LEAF.Vertex[neighbors[1],:] - \
                    self.LEAF.Vertex[hullstarts[0],:]
                if np.cross(a,b) < 0: #order is clockwise
                    outer = [neighbors[0], hullstarts[0], neighbors[1]]
                else:
                    outer = [neighbors[1], hullstarts[0], neighbors[0]]

                #Find clockwise oriented loop around the entire subgraph
                next = find_left_node(sg, outer[-2], outer[-1])
                while next != outer[0]:
                    outer.append(next)
                    next = find_left_node(sg, outer[-2], outer[-1])
                return outer

            self.to_networkx(thr=thr)

            #Find indices of nodes in the cycle basis, not necessarily minimal

            cycles = nx.cycle_basis(self.G)
            if len(cycles) == 0:
                self.cycles = []
                return self.cycles
            keep = [node for cycle in cycles for node in cycle]

            #Get rid of every non-cycle node
            G2 = self.G.copy()
            G2.remove_nodes_from(np.delete(np.arange(self.verts),
                                 np.unique(keep)))

            ''' This algorithm fails when a single node is shared between
            two exterior loops, as in:
                        /\/\
                        \/\/
            '''

            self.cycles = []
            components = list(nx.connected_components(G2))

            #Find isolated cycles first, remove them from G2 before proceeding
            for component in components:
                if np.all([G2.degree(n) == 2 for n in component]):
                    start = list(component)[0]
                    loop = find_loop(G2, start, next(G2.neighbors(start)))
                    self.cycles.append(loop)
                    G2.remove_nodes_from(component)
                    components = list(nx.connected_components(G2))

            while len(components) > 0:
                #find nodes on the exterior of the graph using a convex hull)
                sg = G2.subgraph(components[0])
                ext = find_exterior(sg)

                #start at an exterior node of degree 3, find a loop
                start = [n for n in ext if sg.degree(n) > 2][0]
                startinext = ext.index(start)
                if startinext == 0: end = ext[-1]
                else: end = ext[startinext-1]

                loop = find_loop(sg, start, end)
                self.cycles.append(loop)

                #remove every exterior node and edge that is part of that loop
                remnodes = [n for n in loop if n in ext and G2.degree(n) == 2]
                extedges = [(ext[i], ext[i+1]) for i in range(len(ext)-1)]
                extedges.append((ext[-1], ext[0]))
                loopedges = [(loop[i+1], loop[i]) for i in range(len(loop)-1)]
                loopedges.append((loop[0], loop[-1]))
                remedges = [e for e in loopedges if e in extedges]

                G2.remove_nodes_from(remnodes)
                G2.remove_edges_from(remedges)

                #trim any trees that appear
                degs = list(G2.degree(G2.nodes()))
                degcheck = [d[1] == 1 for d in degs]
                while np.any(degcheck):
                    G2.remove_nodes_from([degs[i][0] for i in np.where(degcheck)[0]])
                    degs = list(G2.degree(G2.nodes()))
                    degcheck = [d[1] == 1 for d in degs]

                #check for isolated loops
                components = list(nx.connected_components(G2))
                for component in components:
                    if np.all([G2.degree(n) == 2 for n in component]):
                        start = list(component)[0]
                        loop = find_loop(G2, start, next(G2.neighbors(start)))
                        self.cycles.append(loop)
                        G2.remove_nodes_from(component)
                        components = list(nx.connected_components(G2))

            return self.cycles

            '''for component in components:
                #find nodes on the exterior of the graph using a convex hull)
                sg = G2.subgraph(component)
                t = time.time()
                hull = ConvexHull(self.LEAF.Vertex[sg.nodes(),:])
                hullverts = np.array(sg.nodes())[hull.vertices]
                print(time.time()-t)
                hullstarts = [n for n in hullverts if sg.degree(n) == 2]

                #unable to handle loops with no convex hull points of deg 2
                if len(hullstarts) == 0:
                    print('exterior loop search failed')
                    raise IndexError('loop has no exterior nodes of deg 2')

                neighbors = list(sg.neighbors(hullstarts[0]))
                #cross product negative if sequence is going clockwise
                a = self.LEAF.Vertex[hullstarts[0],:] - \
                    self.LEAF.Vertex[neighbors[0],:]
                b = self.LEAF.Vertex[neighbors[1],:] - \
                    self.LEAF.Vertex[hullstarts[0],:]
                if np.cross(a,b) < 0: #order is clockwise
                    outer = [neighbors[0], hullstarts[0], neighbors[1]]
                else:
                    outer = [neighbors[1], hullstarts[0], neighbors[0]]

                #Find clockwise oriented loop around the entire subgraph
                next = find_left_node(sg, outer[-2], outer[-1])
                while next != outer[0]:
                    outer.append(next)
                    next = find_left_node(sg, outer[-2], outer[-1])

                #Begin loop searches from nodes of degree > 2
                starts = [n for n in sg.nodes() if sg.degree(n) > 2]
                for start in starts:
                    neighbors = list(sg.neighbors(start))

                    #if start is on the outer loop, remove the clockwise search
                    if start in outer:
                        i = outer.index(start)
                        if i != len(outer)-1:
                            neighbors.remove(outer[i+1])
                        elif i == len(outer)-1:
                            neighbors.remove(outer[0])

                    for n in neighbors:
                        c = [start, n]
                        next = find_left_node(sg, c[-2], c[-1])
                        while next != c[0]:
                            c.append(next)
                            next = find_left_node(sg, c[-2], c[-1])

                        #Assert that the loop contains no nodes
                        poly = Polygon(self.LEAF.Vertex[c,:])
                        try:
                            assert not np.any([poly.contains(Point(
                                self.LEAF.Vertex[n,:])) for n in sg.nodes()])
                            if set(c) not in cyclesets:
                                cyclesets.append(set(c))
                                new_cycles.append(c)
                        except AssertionError:
                            print("overlapping lines made loop search fail")

            self.cycles = new_cycles
            return self.cycles'''

    def n_cycles(self, thr=1e-6):
        if hasattr(self, 'cycles'):
            return len(self.cycles)
        else:
            self.to_networkx(thr=thr)
            cycles = nx.cycle_basis(self.G)
            return len(cycles)

    def count_per_loop(self, type='sinks', thr=1e-4):
        """Counts the number of special nodes (sources, sinks, etc) inside
            each loop
        """
        self.find_cycles(thr=thr)

        counts = np.zeros(len(self.cycles))
        polygons = [Polygon(self.LEAF.Vertex[cycle,:]) for cycle in self.cycles]
        if type == 'sinks':
            inds = self.sinks
        elif type == 'basins':
            inds = self.basins
        elif type == 'source':
            inds = self.sources
        else:
            raise TypeError(type + ' not supported')

        for i in inds:
            loops = []
            inloop = False
            for p in range(len(polygons)):
                if polygons[p].intersects(Point(self.LEAF.Vertex[i,:])):
                    loops.append(p)
                    inloop = True
            if inloop:
                counts[loops] += 1/len(loops)

        return counts

    def tree_lengths(self):
        cycles = self.find_cycles()
        if len(cycles) == 0:
            return None
        polygons = [Polygon(self.LEAF.Vertex[cycle,:]) for cycle in cycles]
        lengths = np.zeros(len(cycles))

        G2 = self.G.copy()

        #Find end nodes, assert no isolated points
        node_list = np.array(G2.nodes())
        degs = np.sum(nx.adjacency_matrix(G2, node_list, weight=None),
                           axis=1)
        assert not np.any(degs == 0)
        assert nx.number_of_selfloops(G2) == 0

        ends = node_list[np.where(degs == 1)[0]]

        while len(ends) > 0:
            for i in ends:
                for p in range(len(polygons)):
                    if polygons[p].intersects(Point(self.LEAF.Vertex[i,:])):
                        lengths[p] += np.linalg.norm(self.LEAF.Vertex[i,:] -
                            self.LEAF.Vertex[next(G2.neighbors(i)),:])
                        break

            G2.remove_nodes_from(ends)

            node_list = np.array(G2.nodes())

            degs = np.sum(nx.adjacency_matrix(G2, node_list, weight=None),
                          axis=1)
            ends = node_list[np.where(degs == 1)[0]]

        return lengths

    def loop_perimeters(self):
        self.find_cycles()
        perimeters = [Polygon(self.LEAF.Vertex[cycle,:]).length for cycle in
                      self.cycles]
        return np.array(perimeters)

    def loop_qhull_perimeters(self):
        self.find_cycles()
        lengths = [Polygon(self.LEAF.Vertex[cycle,:]).convex_hull.length for
                   cycle in self.cycles]
        return np.array(lengths)

    def loop_areas(self, thr=1e-4):
        self.find_cycles(thr=thr)
        areas = [Polygon(self.LEAF.Vertex[cycle,:]).area for cycle in
                 self.cycles]
        return np.array(areas)

    def loop_qhull_areas(self):
        self.find_cycles()
        areas = [Polygon(self.LEAF.Vertex[cycle,:]).convex_hull.area for cycle
                 in self.cycles]
        return np.array(areas)

    def loop_resistance(self,):
        self.find_cycles()
        CM = self.C_matrix(self.C)
        res = []
        for cycle in self.cycles:
            r = 0
            for i in range(len(cycle)-1):
                r += 1/CM[cycle[i], cycle[i+1]]
            res.append(r)
        return np.array(res)

    def loop_avg_edge_resistance(self,):
        res = self.loop_resistance()
        lens = np.array([len(cycle) for cycle in self.cycles])
        return res / lens

    def loop_max_edge_resistance(self,):
        self.find_cycles()
        CM = self.C_matrix(self.C)
        mres = np.zeros(len(self.cycles))
        for c in range(len(self.cycles)):
            for i in range(len(self.cycles[c])-1):
                if (1/CM[self.cycles[c][i], self.cycles[c][i+1]]) > mres[c]:
                    mres[c] = 1/CM[self.cycles[c][i], self.cycles[c][i+1]]
        return mres

    def resistance_distance_mx(self, thr=0):
        #invR = scipy.sparse.linalg.inv(self.R_matrix_sparse(CM))
        G = self.G_matrix_sparse(self.C_matrix_sparse(self.C))
        invR = np.linalg.pinv(G.todense())

        Reff = np.zeros(invR.shape)
        i, j = np.triu_indices(self.verts)
        Reff[i,j] = invR[i,i] + invR[j,j] - invR[i,j] - invR[j,i]

        Reff += Reff.T
        x = np.amin(Reff[Reff > 0])

        Reff[Reff < x] = x
        return Reff

    def resistance_distances(self,):
        Reff = self.resistance_distance_mx()
        return Reff[self.sources, self.sinks], \
            Reff[np.ix_(self.basins, self.sinks)]

    def remove_trees(self, thr=1e-4):
        """ Warning: this trashes the xylem object it's used on
        """
        degs = self.degrees(thr=thr)

        while np.sum(degs == 1) > 0:
            rem = np.argwhere(degs == 1)
            self.LEAF.Vertex = np.delete(self.LEAF.Vertex, rem, axis=0)

            edges = np.argwhere(np.any(np.isin(self.LEAF.Bond, rem),axis=1))
            self.LEAF.Bond = np.delete(self.LEAF.Bond, edges, axis=0)

            for r in rem[::-1]:
                self.LEAF.Bond = np.where(self.LEAF.Bond > r, self.LEAF.Bond-1,
                             self.LEAF.Bond)

            self.C = np.delete(self.C, edges)

            self.verts = self.LEAF.Vertex.shape[0]
            self.bonds = self.LEAF.Bond.shape[0]

            degs = self.degrees(thr=thr)

    def remove_trees_nx(self, thr=1e-4):
        self.to_networkx(thr=thr)

        G2 = self.G.copy()
        node_list = np.array(G2.nodes)
        degrees = np.sum(nx.adjacency_matrix(G2, node_list, weight=None),
                         axis=1)
        remove = node_list[np.where(degrees <= 1)[0]]

        while len(remove) > 0:
            G2.remove_nodes_from(remove)
            node_list = np.array(G2.nodes)
            degrees = np.sum(nx.adjacency_matrix(G2, node_list,
                             weight=None), axis=1)
            remove = node_list[np.where(degrees <= 1)[0]]

        self.G = G2

    def tree_fraction(self, len_attr='length', thr=1e-4):
        """ Return the ratio of edges that are part of a cycle to all edges
        """
        G = self.to_networkx(thr=thr)

        cy = nx.minimum_cycle_basis(G)
        cy_edges = []
        for cycle in cy:
         for i in range(len(cycle) - 1):
             cy_edges.append(tuple(sorted((cycle[i], cycle[i+1]))))

         cy_edges.append(tuple(sorted((cycle[0], cycle[-1]))))

        cy_edges = set(cy_edges)

        cycle_lengths = [G[u][v][len_attr] for u, v in cy_edges]
        all_lengths = [d[len_attr] for u, v, d in G.edges(data=True)]

        return sum(cycle_lengths), sum(all_lengths)

    def path_edges(self, start=None, end=None, dir=False):
        ''' Count the number of paths from start to end(s) each edge is
            involved in
        '''
        if start is None:
            start = self.sources[0]
        if end is None:
            end = self.sinks
        if dir:
            G = self.to_networkx(graph='dir')
        else:
            G = self.to_networkx()
        paths = nx.algorithms.all_simple_paths(G, start, end)
        counts = np.zeros(self.LEAF.Bond.shape[0])

        ''' This way of counting edges can fail in the very narrow case that
        a path goes around a loop that is closed by a single remaining edge, as
        below:
        start ----->------v
        -->  |            |
          end -----<------v

        This is just going to be one that I have to live with for now
        '''
        for path in paths:
            counts += np.all(np.isin(self.LEAF.Bond, path), axis=1)
        return counts

    def path_nodes(self, start=None, end=None, thr=1e-4, dir=False):
        ''' Count the number of paths from start that end at end(s)
        '''
        if start is None:
            start = self.sources[0]
        if end is None:
            end = self.sinks
        if dir:
            G = self.to_networkx(thr=thr, graph='dir')
        else:
            G = self.to_networkx(thr=thr, )
        try:
            paths = nx.algorithms.all_simple_paths(G, start, end)
        except nx.exception.NodeNotFound:
            paths = []
        counts = np.zeros(len(end))
        for path in paths:
            counts[np.where(end == path[-1])] += 1
        return counts[counts > 0]

    def smooth(self, thr=1e-4):
        #self.to_networkx(thr=thr, graph='multi')
        assert type(self.G) == nx.classes.multigraph.MultiGraph

        pos = nx.get_node_attributes(self.G, 'pos')

        c = nx.get_edge_attributes(self.G, 'conductivity')
        l = nx.get_edge_attributes(self.G, 'length')
        w = {e: c[e]**(self.gamma/2)*l[e] for e in self.G.edges(keys=True)}
        nx.set_edge_attributes(self.G, w, 'area')

        for n in [x[0] for x in self.G.degree() if x[1] == 2]:
            neigh = [x for x in self.G[n]]
            if len(neigh) == 1:
                continue
            dic = self.G[n]
            # compute effective conductance of smoothed edge
            l = dic[neigh[0]][0]['length'] + dic[neigh[1]][0]['length']
            #l = np.linalg.norm(
            #    self.G.nodes[neigh[0]]['pos'] - self.G.nodes[neigh[1]]['pos'])
            c = l/(dic[neigh[0]][0]['length']/dic[neigh[0]][0]['conductivity']+\
                dic[neigh[1]][0]['length']/dic[neigh[1]][0]['conductivity'])
            a = dic[neigh[0]][0]['area'] + dic[neigh[1]][0]['area']
            self.G.add_edge(*neigh, conductivity=c, weight=l, length=l, area=a)
            self.G.remove_edge(n, neigh[0])
            self.G.remove_edge(n, neigh[1])
        self.G.remove_nodes_from(
            [x[0] for x in self.G.degree() if x[1] == 0])

    #currently not working
    def top_number_alternative_paths(self, thr=1e-4):
        """
        Computes the number of alternative paths (Nap) in the combinatorics sense
        from the Apex to each of the shoreline outlets.
        """
        apexid = self.sources[0]
        outlets = np.array(self.sinks)

        A = self.adjacency_matrix_asym(thr=thr).T
        epsilon = 10**-15

        # To compute Nap we need to find the null space of L==I*-A', where I* is
        # the Identity matrix with zeros for the diagonal entries that correspond
        # to the outlets.
        D = np.ones((A.shape[0],1))
        D[outlets] = 0
        L = np.diag(np.squeeze(D)) - A
        d, v = np.linalg.eig(L)
        d = np.abs(d)
        null_space_v = np.where(np.logical_and(d < epsilon, d > -epsilon))[0]
        print(len(null_space_v))
        print(len(self.sinks))

        # Renormalize eigenvectors of the null space to have one at the outlet entry
        vN = np.abs(v[:, null_space_v])
        paths = np.empty((len(null_space_v),2))
        for i in range(vN.shape[1]):
            I = np.where(vN[outlets, i] > epsilon)[0]
            print(I)
            print(vN[outlets[I], i])
            vN[:,i] = vN[:,i] / vN[outlets[I], i]
            paths[i,0] = outlets[I]
            paths[i,1] = vN[apexid, i]

        return paths

    # plotting
    def plot(self, style='pipes', thr=1e-4, drawspecial=True, nodelabels=False,
             c=[0,0,0], showscale=False, showbounds=False, magn=8, alpha=False,
             ds=None, cmap='plasma', p=None, v=None, ax=None):
        """Plot the network after simulating to draw it as a graph or as
        a network with conductances.
        Parameters:
            thr: Threshold conductance. Real edges are those with C > thr
            style:
            c: 3-element list specifying pipe color in CMY format
        """
        if ax == None:
            ax = plt.gca()

        if style == 'pipes':
            self.plot_conductivities_raw(magn=magn,
                                         process=lambda x:
                                            (x/x.max())**(0.5*self.gamma),
                                         col=np.array(c),
                                         alpha=alpha,
                                         ax=ax)
        elif style == 'loops':
            self.plot('pipes',
                      drawspecial=drawspecial,
                      nodelabels=nodelabels,
                      showscale=showscale,
                      showbounds=showbounds,
                      magn=magn,
                      alpha=alpha,
                      ax=ax)
            self.drawloops(thr=thr, ax=ax, c=c)
        elif style == 'sticks':
            G = self.to_networkx(thr=thr)
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw_networkx(G, pos, with_labels=False, node_size=6,
                             node_color='black', ax=ax)
        elif style == 'arrows':
            G = self.to_networkx(thr=thr, dir=True)
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw_networkx(G, pos, with_labels=False, node_size=6,
                             node_color='black', ax=ax)
        elif style == 'paths':
            npaths = self.path_edges(dir=True)
            c = plt.get_cmap('magma')
            self.plot(c=c(npaths/max(npaths))[:,:-1],
                drawspecial=False, alpha=False, ax=ax)

            norm = Normalize(vmin=0,vmax=max(npaths))
            sm = plt.cm.ScalarMappable(cmap=c, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label='Number of paths containing link')
        elif style == 'flows':
            if ds == None:
                self.ds = self.fluctuation_ensemble()
                ds = self.ds[:,0]
            self.flow_flux_en_weights()

            CM = self.C_matrix_sparse(self.C)
            G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

            p = scipy.sparse.linalg.spsolve(G, ds[1:])
            Q = (self.C*self.flux_wts)*self.I_mat_red.dot(p)

            start = np.where(Q <= 0, self.LEAF.Bond[:,0], self.LEAF.Bond[:,1])
            end = np.where(Q > 0, self.LEAF.Bond[:, 0], self.LEAF.Bond[:,1])

            x = self.LEAF.Vertex[start,0]
            y = self.LEAF.Vertex[start,1]
            u = self.LEAF.Vertex[end,0] - self.LEAF.Vertex[start,0]
            v = self.LEAF.Vertex[end,1] - self.LEAF.Vertex[start,1]

            colors = np.zeros((len(self.LEAF.Bond[:,0]),4))
            colors[:,:3] = c
            colors[:,3] = (abs(Q) > thr).astype('int')

            linewidths = abs(Q)/100
            head_widths = 0.02*np.ones_like(Q)
            head_lengths = np.where(1.5*head_widths > (u**2+v**2)**0.5,
                                    (u**2+v**2)**0.5,
                                    1.5*head_widths)

            fast = np.nonzero(np.arange(len(x))*colors[:,3])[0]
            for f in fast:
                plt.arrow(x[f], y[f], u[f], v[f], width=linewidths[f],
                          head_width=head_widths[f],
                          head_length=head_lengths[f],
                          length_includes_head=True, color=colors[f,:],
                          zorder=5, ax=ax)
        elif style == 'pressure':
            if p is None:
                if ds is None:
                    self.ds = self.fluctuation_ensemble()
                    ds = self.ds[:,0]
                CM = self.C_matrix_sparse(self.C)
                G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

                ds = self.fluctuation_ensemble()
                p = scipy.sparse.linalg.spsolve(G, ds[1:])
                p -= min(p)
                p /= max(p)
                if v is None:
                    v = (np.amin(p), np.amax(p))

            from matplotlib.collections import PolyCollection
            cmap = cm.get_cmap(cmap)
            facecolors = cmap((p-v[0])/v[1])

            #plt.scatter(self.LEAF.Vertex[1:,0], self.LEAF.Vertex[1:,1],
            #            c=facecolors)

            vor = self.LEAF.Voronoi

            rem = self.LEAF.RestVertices

            polys = []
            plottinginds = np.delete(np.arange(vor.points.shape[0]),
                                     rem)
            #plottinginds = np.arange(vor.points.shape[0])
            ##problem = np.where(np.all(vor.points == self.LEAF.Vertex[0,:],
            #                          axis=1))[0]
            plottinginds = np.delete(plottinginds, 0)

            for regionind in vor.point_region[plottinginds]:
                region = np.array(vor.regions[regionind])
                if -1 not in region:
                    polys.append(vor.vertices[region,:])
                else:
                    a = region[region >= 0]
                    polys.append(vor.vertices[a,:])

            ax.add_collection(PolyCollection(polys,
                                             facecolors=facecolors,
                                             alpha=1))

        if drawspecial:
            self.drawspecial(ax=ax)

        if nodelabels:
            bonds = self.LEAF.Bond[self.C > thr, :]
            nodes = np.unique(bonds.flatten())
            for n in nodes:
                ax.text(self.LEAF.Vertex[n,0], self.LEAF.Vertex[n,1], str(n))

        if showscale:
            if self.LSHP.comment == 'circle' or self.LSHP.comment == 'hexagon':
                x = [-0.8, -0.8 + self.sigma]
                ax.plot(x, [-1.05]*2, 'k', lw=2)
                ax.set_ylim([-1.08,1])
                ax.text(np.mean(x) - 0.03, -1.14, '$\sigma$', fontsize=20)
            elif self.LSHP.comment == 'square' or \
                self.LSHP.comment == 'triangle':
                x = [-0.8, -0.8 + self.sigma]
                ax.plot(x, [1.1]*2, 'k', lw=2)
                ax.text(np.mean(x) - 0.03, 1.04, '$\sigma$')
        else:
            self.scale_plt_figure(ax=ax)

        if showbounds:
            ax.plot(self.LSHP.polyedge[:,0], self.LSHP.polyedge[:,1], 'b',
                     alpha=0.4)

    def drawloops(self, thr=1e-6, ax=None, c=[0,0,1]):
        self.find_cycles(thr=thr)
        if ax == None:
            ax = plt.gca()
        for cycle in self.cycles:
            #facecolor = np.random.rand(3)
            #if np.all(facecolor > 0.5):
            #    facecolor[np.random.choice(3)] = np.random.rand()*0.5
            p = pltPolygon(self.LEAF.Vertex[cycle,:], facecolor=c,
                alpha=0.5)
            ax.add_patch(p)

    def drawspecial(self, ax=None):
        """Plot the nodes with specified boundary conditions"""
        if ax == None:
            ax = plt.gca()
        ax.plot(self.LEAF.Vertex[self.sinks,0],
                 self.LEAF.Vertex[self.sinks,1], 'y.', alpha=1,
                 markersize=12)
        if self.sources is not None:
            ax.plot(self.LEAF.Vertex[self.sources,0],
                     self.LEAF.Vertex[self.sources,1], 'r.', alpha=1,
                     markersize=15)

    def color_plot(self, mode='currents'):
        CM = self.C_matrix_sparse(self.C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

        if mode == 'current':
            C = self.C.copy() * self.flux_wts
            colors = np.abs(np.sum((C[:,np.newaxis]*self.I_mat_red.dot(
                scipy.sparse.linalg.spsolve(G, self.ds[1:,:]))), axis=1) / self.ds.shape[1])
        elif mode == 'pressure':
            colors = np.abs(np.sum((self.I_mat_red.dot(
                scipy.sparse.linalg.spsolve(G, self.ds[1:,:]))), axis=1) / self.ds.shape[1])

        colors = colors[np.where(abs(colors) < 5)]

        colors -= min(colors)

        colors /= max(colors)

        rainbow = cm.get_cmap('Greys')

        xs, ys = self.bond_coords()
        conds = (self.C.copy() / max(self.C))**0.25

        colors = rainbow(colors)
        #colors[:,3] = conds

        segs = np.array([np.array([xs[bond], ys[bond]]).T for bond in range(len(ys))])
        '''#Thresholding to plot only lines with C > thr
        a = segs[np.where(self.C.copy()>thr)]
        acond = conds[np.where(self.C.copy()>thr)]
        acolors = colors[np.where(self.C.copy()>thr)]'''

        ax = plt.gca()

        line_segments = LineCollection(segs,
                               linewidths=0.2,
                               colors=colors,
                               capstyle='round',
                               cmap='rainbow')
        ax.add_collection(line_segments)

        line_segments = LineCollection(segs,
                               linewidths=conds*10,
                               colors=colors,
                               capstyle='round',
                               cmap='rainbow')
        ax.add_collection(line_segments)

        self.scale_plt_figure(ax=ax)

        self.drawspecial()

    #storage
    def to_networkx(self, thr=1e-4, graph=None):
        """ Return a NetworkX graph representing this network
        """
        graphs = {None: nx.classes.graph.Graph,
                  'dir': nx.classes.digraph.DiGraph,
                  'multi': nx.classes.multigraph.MultiGraph}
        if hasattr(self, 'G'):
            if graphs[graph] != type(self.G):
                pass
            else:
                return self.G

        if hasattr(self,'flow_wts'):
            pass
        else:
            self.flow_flux_en_weights()

        if graph == 'dir':
            G = nx.DiGraph()

            self.ds = self.fluctuation_ensemble()
            self.flow_flux_en_weights()

            CM = self.C_matrix_sparse(self.C)
            L = self.G_matrix_sparse(CM)[1:,1:].tocsc()

            # choose ensemble mode 1 just for simplicity
            # p = scipy.sparse.linalg.spsolve(L, self.ds[1:,20])
            # Q = (self.C*self.flux_wts)*self.I_mat_red.dot(p)
            # self.LEAF.Bond[Q >= 0, :] = self.LEAF.Bond[Q >= 0, ::-1]
        elif graph == 'multi':
            G = nx.MultiGraph()
        else:
            G = nx.Graph()

        eds = self.C > thr

        # Remove sinks for loop counting
        if self.system == 'delta':
            remove = np.all(np.isin(self.LEAF.Bond, self.sinks), axis=1)
            eds = np.logical_and(eds, np.logical_not(remove))

        for (a, b), c, w, l in zip(self.LEAF.Bond[eds,:],
                self.C[eds], self.flow_wts[eds], self.bond_lens[eds]):
            G.add_edge(a, b, conductivity=c, weight=1./w, length=l)
            G.nodes[a]['pos'] = self.LEAF.Vertex[a,:]
            G.nodes[b]['pos'] = self.LEAF.Vertex[b,:]

        if nx.number_of_selfloops(G) > 0:
            G.remove_edges_from(nx.selfloop_edges(G))

        self.G = G
        return self.G

    def from_networkx(self,): #not done yet
        """ Not implemented yet """
        return None

    def save(self, name):
        """ Returns a dict that can be used to reconstruct the
        XylemNetwork object using the from_dict method but contains only
        bare bones data to save memory.
        """
        data = {}

        data['type'] = self.system
        data['constructor_params'] = self.constructor_params
        data['attributes'] = {
                'C': self.C,
                }

        if name[-2:] != '.p':
            name += '.p'

        with open(name, 'wb') as f:
            pickle.dump(data, f)

        return data

class DeltaNetwork(NetworkSuite):
    """ An object built upon the NetworkSuite object for the simulation of
        river deltas and marshes
    """
    # initialization and simulation
    def __init__(self, LEAF, LSHP, size, cst, C0=None,
                 fluctuations='river',
                 inputs='line_source_wide', n_sources=1, sourceinds=None,
                 outputs='line', n_sinks=25, sink_fraction=None, sinkinds=None,
                 basins='random', basin_fraction=None, basininds=None):

        super(DeltaNetwork, self).__init__(LEAF, LSHP, size,
            C0=C0,
            inputs=inputs, n_sources=n_sources, sourceinds=sourceinds,
            outputs=outputs, n_sinks=n_sinks, sink_fraction=sink_fraction,
            sinkinds=sinkinds)

        self.system = 'delta'
        self.fluctuations = fluctuations

        self.gamma = 2/3 #0.666 for rivers, from exp data
        self.cst = cst

        self.set_basins(basins, basin_fraction, basininds)

        self.constructor_params = {
                'LEAF': LEAF,
                'LSHP': LSHP,
                'size': size,
                'cst': cst,
                'C0': C0,
                'fluctuations': self.fluctuations,
                'sources': self.sources,
                'sinks': self.sinks,
                'basins': self.basins,
                'basin_fraction': basin_fraction
                }

    def __repr__(self):
        return 'Geography object with T = ' + str(self.cst)

    def set_basins(self, basins, basin_fraction, basininds):
        if basin_fraction == None and basininds == None:
            self.basins = []
            return self.basins

        if basins == 'generate':
            bondnum = np.bincount(self.LEAF.Bond.flatten())
            ends = np.where(bondnum == 1)[0]
            self.basins = [x for x in ends if x not in self.sinks and
                           x not in self.sources]
            return self.basins

        if basininds is not None:
            self.basins = basininds
            return self.basins

        n_basins = int(basin_fraction*self.verts)
        if basins == 'random':
            #random basins
            p = np.ones(self.verts)
            p[self.sources] = 0
            p[self.sinks] = 0
            p /= np.sum(p)
            self.basins = np.random.choice(self.verts,
                size=n_basins, p=p, replace=False)
            return self.basins
        if basins == 'square' or basins == 'triangle':
            extended_basins = int(n_basins*4/self.area)
            # Lattice between +/-0.98 to limit intersecting lines on
            # outside region of graph
            x, y = np.meshgrid(
                np.linspace(-0.98,0.98,int(np.sqrt(extended_basins))),
                np.linspace(-0.98,0.98,int(np.sqrt(extended_basins))))

            if basins == 'triangle':
                x[::2, :] += (x[0,1] - x[0,0])/2

            A = np.array([x.flatten(),y.flatten()]).T
        elif basins == 'linear':
            #this feature hasn't been tested!
            #basins distributed in y by to linear power law distribution
            x = np.linspace(-1,1, n_basins)
            y = -2 * np.random.power(2, size=n_basins) + 1

            A = np.array([x,y]).T

        self.basins = []
        poly = Polygon(self.LSHP.polyedge)
        for i in range(A.shape[0]):
            if poly.intersects(Point(A[i,:])):
                distsq = (self.LEAF.Vertex[:,0] - A[i,0])**2 + \
                          (self.LEAF.Vertex[:,1] - A[i,1])**2
                bas = np.argmin(distsq)
                if bas not in self.sources and bas not in self.sinks \
                    and bas not in self.basins:
                    self.LEAF.Vertex[bas,:] = A[i,:]
                    self.basins.append(bas)

        self.bond_lens = self.bond_lengths()
        scale = self.bond_lens.max()

        self.bond_lens /= scale

    def fluctuation_ensemble(self, fluctuations=None):
        resolution = 30

        ds = np.zeros((self.verts, resolution))

        nonsinks = np.delete(range(self.verts), self.sinks)
        ds[nonsinks, :] = 10**(-7)

        if fluctuations == None:
            fluctuations = self.fluctuations

        if fluctuations == 'river':
            ''' Note: you only need to do half the sin period since the
            other half has the same information'''

            tides = np.cos(np.linspace(0, np.pi, resolution, endpoint=True))

            ''' # dec19rivers, jan20rivers, may20rivers
            ds[self.sources,:] = 1/self.n_sources
            ds[self.basins,:] = self.cst*tides/len(self.basins)
            ds[self.sinks,:] = -np.sum(ds,axis=0)/self.n_sinks

            ds /= -1*np.sum(ds.clip(min=0),axis=0) # normalize inputs'''

            # jun20rivers
            ds[self.sources,:] = 1/self.n_sources
            ds[self.basins,:] = (self.cst) * tides / len(self.basins)
            ds /= np.sum(ds[:,0])
            ds[self.sinks, :] = -np.sum(ds, axis=0) / self.n_sinks
        elif fluctuations == 'marsh':
            '''#Basins fluctuating at spatially-random phase
            basins = np.random.permutation(self.basins)
            rows = np.tile(np.linspace(0, 2*np.pi, resolution, endpoint=False),
                           (len(self.basins),1))
            cols = np.tile(np.linspace(0, 2*np.pi, len(self.basins),
                           endpoint=False)[:,np.newaxis], (1,resolution))
            ds[self.basins, :] = np.sin(cols+rows)'''

            '''L = max(self.LSHP.polyedge[:,1]) - min(self.LSHP.polyedge[:,1])
            y = self.LEAF.Vertex[self.basins,1]

            phi = np.pi

            cols = np.tile(phi*y[:,np.newaxis]/L, (1,resolution))
            rows = np.tile(np.linspace(0, phi, resolution,
                           endpoint=False), (len(self.basins),1))

            ds[self.basins,:] = np.sin(cols+rows)
            ds[regular_nodes, :] = 10**(-7)
            ds[self.sinks, :] -= ds.sum(axis=0)/len(self.sinks)'''

            tides = np.cos(np.linspace(0, np.pi, resolution, endpoint=True))

            inds = np.random.permutation(len(self.basins))
            rows = np.tile(np.linspace(0, 2*np.pi, resolution, endpoint=False),
                           (len(self.basins),1))
            cols = np.tile(np.linspace(0, 2*np.pi, len(self.basins),
                           endpoint=False)[:,np.newaxis], (1,resolution))

            ds[self.sources,:] = 1/self.n_sources
            ds[self.basins,:] = (self.cst) * tides / len(self.basins) + \
                self.noise*self.cst*np.cos(rows+cols)[inds,:]/len(self.basins)
            ds /= np.sum(ds[:,0])
            ds[self.sinks, :] = -np.sum(ds, axis=0) / self.n_sinks

        elif fluctuations == 'entropy':
            ''' The first column is source only, remaining columns are basins
                only.
            '''
            tides = np.cos(np.linspace(0, np.pi, resolution-1, endpoint=True))

            ds[self.sources, 0] = 1/self.n_sources
            ds[self.basins, 0] = 10**(-7)
            ds[self.sources, 1:] = 10**(-7)
            ds[self.basins, 1:] = (self.cst) * tides / len(self.basins)
            ds[self.sinks, :] = -np.sum(ds, axis=0) / self.n_sinks

            '''ds[self.sources, 1:] = 10**(-7)
            ds[self.basins, 1:] = np.cos(np.linspace(0, np.pi, resolution-1,
                endpoint=True))
            ds[self.sinks, :] = -np.sum(ds,axis=0)/self.n_sinks

            ds /= np.sum(ds.clip(min=0),axis=0)'''

        #print(np.sum(ds, axis=0))

        return ds

    @classmethod
    def load(cls, name):
        """ Takes a data dict containing only the
        bare bones data of the network and turns it into
        a fully fledged XylemNetwork object.
        """

        if name[-2:] != '.p':
            name += '.p'
        with open(name, 'rb') as f:
            saved = pickle.load(f)

        assert saved['type'] == 'delta', 'must be loading a delta network'

        data = saved['constructor_params']

        de = cls(data['LEAF'], data['LSHP'],
                data['size'], data['cst'],
                C0=data['C0'], fluctuations=data['fluctuations'],
                sourceinds=data['sources'],
                sinkinds=data['sinks'],
                basininds=data['basins'], basin_fraction=data['basin_fraction'],)

        de.__dict__.update(saved['attributes'])

        return de

    @classmethod
    def make_river(cls, c, density=65, basin_fraction=0.15, shape='square',
            basins='triangle', n_sources=1, n_sinks=29):
        if shape == 'circle':
            trimming_percentile = 100
            outputs = 'semicircle'
            inputs = 'line_source'
        else:
            outputs = 'line'
            trimming_percentile = 99
            if shape in ['square', 'sq', 'strip', 'sine']:
                inputs = 'line_source_wide'
            elif shape == 'triangle' or shape == 'tri':
                inputs = 'line_source'
            else:
                inputs = 'line_source'
        if shape == 'invtriangle':
            outputs = 'invtriangle'
        LSHP = NetworkSuite.make_LSHP(shape)
        LEAF = NetworkSuite.make_LEAF('River', density, 'random', LSHP,
                            trimming_percentile=trimming_percentile)
        delta = cls(LEAF, LSHP, density, c,
                    fluctuations='river',
                    inputs=inputs, n_sources=n_sources,
                    outputs=outputs, n_sinks=n_sinks,
                    basins=basins, basin_fraction=basin_fraction)

        return delta

    @classmethod
    def make_marsh(cls, c, noise, density=65, basin_fraction=0.15,
                   shape='square', basins='triangle', n_sources=1, n_sinks=29):
        assert noise >= 0 and noise <= 1
        if shape == 'circle':
            trimming_percentile = 100
            outputs = 'semicircle'
            inputs = 'line_source'
        else:
            outputs = 'line'
            trimming_percentile = 99
            if shape in ['square', 'sq', 'strip', 'sine']:
                inputs = 'line_source_wide'
            elif shape == 'triangle' or shape == 'tri':
                inputs = 'line_source'
            else:
                inputs = 'line_source'
        if shape == 'invtriangle':
            outputs = 'invtriangle'
        LSHP = NetworkSuite.make_LSHP(shape)
        LEAF = NetworkSuite.make_LEAF('Marsh', density, 'random',
                            LSHP, trimming_percentile=trimming_percentile)

        marsh = cls(LEAF, LSHP, density, c,
                    fluctuations='marsh',
                    inputs=inputs, n_sources=n_sources,
                    outputs=outputs, n_sinks=n_sinks,
                    basins=basins, basin_fraction=basin_fraction)

        marsh.noise = noise

        return marsh

    def simulate(self, plot=False, movie_dir=None, entropy=False):
        print('\nGeography simulation with T=%0.2f' % self.cst)
        initialtime = time.time()
        print('Number of sinks: %d' % len(self.sinks))
        print('Number of basins: %d' % len(self.basins))

        self.simulate_base(plot=plot, movie_dir=movie_dir, plot_interval=1,
            entropy=entropy, timesteps=1e5)

        print('Simulation complete')
        print('Runtime: ' + str(round((time.time()-initialtime)/60, 2)) + \
            ' minutes')

    # processing
    def remove_trees_nx(self, thr=1e-4):
        if not hasattr(self, 'G'):
            self.to_networkx(thr=thr)

        G2 = self.G.copy()
        node_list = np.array(G2.nodes)
        degrees = np.sum(nx.adjacency_matrix(G2, node_list, weight=None),
                         axis=1)
        remove = node_list[np.where(degrees <= 1)[0]]
        remove = remove[np.logical_not(np.isin(
            remove, np.append(self.sources, self.sinks)))]

        while len(remove) > 0:
            G2.remove_nodes_from(remove)
            node_list = np.array(G2.nodes)
            degrees = np.sum(nx.adjacency_matrix(G2, node_list,
                             weight=None), axis=1)
            remove = node_list[np.where(degrees <= 1)[0]]
            remove = remove[np.logical_not(np.isin(
                remove, np.append(self.sources, self.sinks)))]
            if self.sinks[0] in remove:
                assert False

        self.G = G2

    def thin(self, thr=1e-4, basethr=1e-8):
        """ Trims internal trees
        """
        self.remove_trees_nx(thr=basethr)

        cs = nx.get_edge_attributes(self.G, 'conductivity')
        self.G.remove_edges_from([key for key in cs.keys() if cs[key] < thr])
        self.G.remove_nodes_from([n[0] for n in self.G.degree if n[1] == 0])

        return self.G

    # statistics
    def flow_change(self, thr=1e-4):
        self.flow_flux_en_weights()
        C = self.C*self.flux_wts
        ds = self.fluctuation_ensemble()

        CM = self.C_matrix_sparse(C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()
        Q = C[:,np.newaxis]*self.I_mat_red.dot(
            scipy.sparse.linalg.spsolve(G, ds[1:,[0,-1]]))

        filter = self.C > thr
        tot = np.sum(filter)

        change = np.where(Q[filter,0] > 0,1,-1)*np.where(Q[filter,1] > 0,1,-1)
        return np.sum(change-1)/-2/tot

    def pressure_covar(self, mode='euclidean'):
        """Calculates covariances in pressure between basins of geography
            simulations by subtracting the mean pressure from all basins
            and calculating all covariances from there
        Parameters:
            mode: either 'euclidean' to return euclidean distance between
                basins or 'ydist' to return y-axis distance between
                basins
        Returns:
            x: list of distances between basins (according to mode)
            y: list of covariances between basins corresponding to list x
        """
        self.ds = self.fluctuation_ensemble()
        CM = self.C_matrix_sparse(self.C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

        p = np.zeros((self.verts, self.ds.shape[1]))
        p[1:,:] = scipy.sparse.linalg.spsolve(G, self.ds[1:,:])
        print(np.amax(p)-np.amin(p))

        basinsp = p[self.basins,:]
        basinsp -= np.mean(basinsp, axis=0) #subtract off average pressures
        pick = np.random.choice(basinsp.shape[0], 1000, replace=False)
        pick = np.sort(pick)[::-1]
        basinsp = basinsp[pick, :]
        c = np.cov(basinsp)

        #can calculate exactly how many elements in x and y based on # of
        #basins but I've been too lazy to do this so far
        u1, u2 = np.triu_indices(c.shape[0], k=1)
        b = np.array(self.basins)[pick]

        #x = np.linalg.norm(self.LEAF.Vertex[b[u1], :] - \
        #    self.LEAF.Vertex[b[u2], :], axis=1)
        x = np.abs(self.LEAF.Vertex[b[u1], 0] - self.LEAF.Vertex[b[u2], 0])
        y = c[u1, u2]
        #pt = np.random.choice(np.where((x > 1.5) & (y > 6e3))[0])

        plt.figure(figsize=(10,8))

        plt.plot(x, y, '.')
        #plt.plot(x[pt], y[pt], '^r')
        #print(x[pt], y[pt])

        #self.plot('pipes')
        #plt.plot(self.LEAF.Vertex[[b[u1[pt]], b[u2[pt]]], 0], self.LEAF.Vertex[[b[u1[pt]], b[u2[pt]]], 1], '^r')
        #plt.show()

        return np.around(np.array(x), decimals=4), np.array(y)

    def binned_pressure_covar(self, mode='euclidean', return_original=False):
        x, y = self.pressure_covar(mode=mode)
        xred = np.unique(x)
        yred = np.zeros(len(xred))
        for i in range(len(xred)):
            yred[i] = np.sum(y[x == xred[i]]) / np.sum(x == xred[i])
        return x, y, xred, yred

    def sliding_covar(self, mode='euclidean', resolution=40):
        x, y = self.pressure_covar(mode=mode)
        centers = np.linspace(0, max(x), resolution+2)
        deltax = centers[1] - centers[0]
        xred = np.zeros(resolution)
        yred = np.zeros(resolution)
        for i in range(resolution):
            xred[i] = centers[i+1]
            yvals = y[np.where((x > centers[i]-deltax) &
                               (x < centers[i]+deltax))[0]]
            yred[i] = np.sum(yvals) / len(yvals)
        return x, y, xred, yred

    def pressure_suite(self, k=7, thr=0, plot=False, mode=None, n=[491]):
        """ k is the column of self.ds that we are looking at
        """
        self.ds = self.fluctuation_ensemble()

        CM = self.C_matrix_sparse(self.C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()
        p = np.zeros((self.verts, self.ds.shape[1]))
        p[1:,:] = scipy.sparse.linalg.spsolve(G, self.ds[1:,:])
        basinsp = p[self.basins,:]

        p = np.concatenate((p, p[:,::-1]), axis=1)
        plt.plot(p[self.sources[0],:])
        plt.plot(p[self.basins[0],:])
        plt.plot(p[self.basins[50],:])
        plt.show()

        for i in n:
            self.plot(style='sticks')

            plt.plot(self.LEAF.Vertex[i,0], self.LEAF.Vertex[i,1], '^r', ms=10)

            inds = np.delete(np.arange(Reff.shape[1]), i)
            c = np.delete(Reff[i,:], i)

            plt.scatter(self.LEAF.Vertex[inds,0], self.LEAF.Vertex[inds,1],
                c=c, cmap=vir, norm=LogNorm(vmin=vmin, vmax=np.amax(Reff)), zorder=-1)
            #plt.colorbar()
            plt.show()

        x = np.linalg.norm(self.LEAF.Vertex[u1, :] - \
            self.LEAF.Vertex[u2,:], axis=1)

    def pressure_suite_new(self, k=7, thr=0, plot=False, mode=None):
        """ k is the column of self.ds that we are looking at
        """

        CM = self.C_matrix_sparse(self.C) #Conductivity matrix

        ResMat = scipy.sparse.coo_matrix((1/self.C, (self.LEAF.Bond[:,0],
                    self.LEAF.Bond[:,1])), shape=(self.verts, self.verts))
        ResMat = ResMat + ResMat.T;
        ResLap = scipy.sparse.spdiags(ResMat.sum(axis=0), [0], ResMat.shape[0],
                    ResMat.shape[0], format='coo') - ResMat
        invR = scipy.sparse.linalg.inv(ResLap)

        G = self.G_matrix_sparse(CM).tocsc()
        #p = scipy.sparse.linalg.spsolve(G, self.ds[:,k])

        invG = scipy.sparse.linalg.inv(G)

        if thr == 0:
            u1, u2 = np.triu_indices(n=invR.shape[0],
                m=invR.shape[1], k=1)
        if mode == 'basins':
            testpts = self.basins
            u = np.array(np.triu_indices(n=invR.shape[0],
                m=invR.shape[1], k=1)).T
            u = u[np.all(np.isin(u, testpts), axis=1), :]
            u1 = u[:,0]
            u2 = u[:,1]
        else:
            testpts = np.unique(self.LEAF.Bond[self.C > thr, :])
            u = np.array(np.triu_indices(n=invR.shape[0],
                m=invR.shape[1], k=1)).T
            u = u[np.all(np.isin(u, testpts), axis=1), :]
            u1 = u[:,0]
            u2 = u[:,1]

        Reff = invR[u1,u1] + invR[u2,u2] - invR[u1,u2] - \
            invR[u2,u1]
        Ceff = invG[u1,u1] + invG[u2,u2] - invG[u1,u2] - invG[u2,u1]

        Reff[Reff < 0] = 1e-25

        self.plot(style='sticks', drawspecial=False)

        plt.plot(self.LEAF.Vertex[self.sources[0],0], self.LEAF.Vertex[self.sources[0],1], '^r', ms=10)

        plt.scatter(self.LEAF.Vertex[testpts,0], self.LEAF.Vertex[testpts,1],
            c=Reff[self.sources[0], :], cmap=vir, norm=LogNorm(vmin=vmin, vmax=np.amax(Reff)), zorder=-1)
        #plt.colorbar()
        plt.show()

        x = np.linalg.norm(self.LEAF.Vertex[u1, :] - \
            self.LEAF.Vertex[u2,:], axis=1)
        #dp = np.abs(p[u1, k] - p[u2, k])
        #Q = dp/Reff
        #Q2 = dp/Ceff

        if plot:
            plt.hexbin(x, Reff.tolist()[0], yscale='log')
            plt.show()

    def graph_entropy(self):
        """ Entropy of the graph at time k based on the description given in:
        DOI: 10.1080/0305215512331328259
        """
        ds = self.fluctuation_ensemble('entropy')

        # Calculate S0
        I = self.sources + self.basins + self.sinks
        p0 = ds[I,:].clip(min=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S0 = -1*np.sum(p0*np.nan_to_num(np.log(p0)),axis=0)

        # Calculate Pn
        CM = self.C_matrix_sparse(self.C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()

        Q = self.C[:,np.newaxis]*self.I_mat_red.dot(
            scipy.sparse.linalg.spsolve(G, ds[1:,:]))

        PS = np.zeros(Q.shape[1])
        for j in [0,1]:
            """ Cheat by making an extra row for the 'super sink' to make
            calculations easier"""
            nodeQ = scipy.sparse.csc_matrix(
                (Q[:,j], (self.LEAF.Bond[:,0], self.LEAF.Bond[:,1])),
                shape=(self.verts+1, self.verts+1)
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nodeQ[self.verts, self.sinks] = ds[self.sinks, j]
                nodeQ[self.verts, self.basins] = ds[self.basins, j]
            nodeQ -= nodeQ.T
            nodeQ[nodeQ < 0] = 0
            T = np.squeeze(np.asarray(np.sum(nodeQ, axis=0)))[:-1]
            p = np.asarray(nodeQ[:, np.where(T != 0)[0]] / T[T != 0])
            '''logp = p
            logp[logp != 0] = np.log(logp[logp != 0])
            print(logp)'''
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S = -1*np.sum(p*np.nan_to_num(np.log(p)), axis=0)

            PS[j] = np.sum(T[T != 0]*S)

        return S0[[0,1]] + PS[[0,1]]

    def flow_entropy(self, Q_matrix):
        """ calculate the flow entropy (path entropy) from
        the flow matrix according to

        doi:10.1103/PhysRevLett.104.048703
        """
        #t0 = time.time()
        # only positive flows
        Qpos = Q_matrix.copy()
        Qpos[Qpos < 0] = 0

        # normalize by total outflows
        outflows = Qpos.sum(axis=1)
        P_matrix = Qpos/outflows[:,np.newaxis]
        P_matrix[np.logical_not(np.isfinite(P_matrix))] = 0

        # total outflow at each node
        P_i = Qpos.sum(axis=0)

        # local entropy
        logs = np.log2(P_matrix)
        logs[np.logical_not(np.isfinite(logs))] = 0

        S_i = np.sum(P_matrix*logs, axis=1)

        #t1 = time.time()

        #print "time:", t1 - t0

        return -np.sum(P_i*S_i)

    def flow_entropies(self, C):
        """ Calculate the mean flow entropy for
        injecting currents at all nodes and removing it at the
        source.
        """
        #t0 = time.time()
        self.flow_flux_en_weights()
        C_weighted = C*self.flow_wts
        CM = self.C_matrix_sparse(C_weighted)
        G = self.G_matrix_sparse(CM)[1:,1:]

        # construct RHS for flows between sink and one
        # particular node
        #t1 = time.time()
        rhs = np.eye(G.shape[0]) # if result is dense then
                              # faster if rhs is also dense
        Qs = C_weighted[:,np.newaxis]*self.I_mat_red.dot(scipy.sparse.linalg.spsolve(G, rhs))

        ents = []
        for i in range(self.verts - 1):
            Qm = np.zeros((self.verts, self.verts))

            for (a, b), Q in zip(self.LEAF.Bond, Qs[:,i]):
                Qm[a,b] = Q
                Qm[b,a] = -Q

            ents.append(self.flow_entropy(Qm))
        #t1 = time.time()

        #print "Time", t1-t0

        return np.array(ents)

    def bridges(self, thr=1e-4, smooth=False, weight=True):
        """ Returns fraction of channel area found in bridges.
        Subtract from 1 to get fraction of channel area found in loops.
        """

        self.thin(thr=thr)
        if smooth:
            self.smooth(thr=thr)

        conductivity = nx.get_edge_attributes(self.G, 'conductivity')
        length = nx.get_edge_attributes(self.G, 'length')

        edges = list(nx.bridges(self.G))

        lb = np.array([length[e] for e in edges])
        l = np.array([length[e] for e in self.G.edges()])

        if weight == True:
            cb = np.array([conductivity[e] for e in edges])
            c = np.array([conductivity[e] for e in self.G.edges()])
            return np.sum(cb**(self.gamma/2) * lb) / \
                np.sum(c**(self.gamma/2) * l)
        elif weight == 'new':
            cb = np.array([conductivity[e] for e in edges])
            return np.sum(cb**(self.gamma/2) * lb) / self.total_area()
        else:
            return np.sum(lb) / np.sum(l)

    def total_area(self, thr=1e-8):
        keep = np.where(self.C >= thr)
        return np.sum(self.bond_lens[keep]*self.C[keep]**(self.gamma/2))

    def loop_ranking(self,):
        self.find_cycles(thr=1e-4)
        x = np.array([np.median(a.C[c]) for c in self.cycles])
        rank = np.argsort(x)[::-1]
        if len(rank) == 0:
            return 1e-4
        if len(rank) > 0 and len(rank) < 5:
            return 1e-4
        if len(rank) >= 5:
            return np.median(a.C[self.cycles[rank[5]]])

    def mstdiff(self, thr=1e-4, weight=True):
        self.to_networkx(thr=thr, graph='multi')
        self.thin(thr=thr)
        self.smooth(thr=thr)

        if weight:
            tree = nx.maximum_spanning_tree(self.G, weight='area')
            a = nx.get_edge_attributes(tree, 'area')
            fulla = nx.get_edge_attributes(self.G, 'area')
        else:
            tree = nx.maximum_spanning_tree(self.G, weight='length')
            a = nx.get_edge_attributes(tree, 'length')
            fulla = nx.get_edge_attributes(self.G, 'length')

        mstarea = np.sum([a[e] for e in tree.edges(keys=True)])
        totalarea = np.sum([fulla[e] for e in self.G.edges(keys=True)])
        return 1 - mstarea / totalarea

    # plotting
    def drawspecial(self, ax=None):
        if ax == None:
            ax = plt.gca()
        ax.plot(self.LEAF.Vertex[self.sinks,0],
                 self.LEAF.Vertex[self.sinks,1], 'y.', alpha=1,
                 markersize=12, axes=ax)
        if self.sources is not None:
            ax.plot(self.LEAF.Vertex[self.sources,0],
                     self.LEAF.Vertex[self.sources,1], 'r.', alpha=1,
                     markersize=15, axes=ax)
        ax.plot(self.LEAF.Vertex[self.basins,0],
                 self.LEAF.Vertex[self.basins,1], 'c.', alpha=0.5,
                 markersize=18, axes=ax)

    def ensembleplot(self, dir):
        self.ds = self.fluctuation_ensemble()

        CM = self.C_matrix_sparse(self.C)
        G = self.G_matrix_sparse(CM)[1:,1:].tocsc()
        p = scipy.sparse.linalg.spsolve(G, self.ds[1:,:])
        v = (np.amin(p), np.amax(p)-np.amin(p))
        p = np.concatenate((p, p[:,::-1]), axis=1)

        t = np.linspace(0, 2*np.pi, p.shape[1])
        tides = np.cos(t)
        t *= p.shape[1]/2/np.pi

        fig = plt.figure(figsize=(15,10))
        grid = plt.GridSpec(4,5, hspace=0.05, wspace=0.1, left=0.1, right=0.9)
        for i in range(p.shape[1]):
            print(i)

            if self.fluctuations == 'river':
                fig.add_subplot(grid[1:3,0])
                plt.plot(t, tides*self.cst)
                plt.plot([t[i]]*2, [min([-self.cst,-1]), max([self.cst,1])],
                    'k')
                plt.plot([0,p.shape[1]], [0,0], 'k--')
                plt.xlabel('Ensemble state')
                plt.ylabel('Total flow through basins relative to river input')

            fig.add_subplot(grid[:,0:])
            self.plot(style='pressure', cmap='cividis', p=p[:,i], v=v,
                drawspecial=False)
            self.plot(style='pipes', drawspecial=False)
            plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                cmap='cividis'), ax=plt.gca(),
                label='Fraction of max potential')
            plt.axis('off')

            plt.savefig(dir+'/%05d.png' % i)
            plt.clf()

def crop_LEAF(LEAF, C0, lengths, x=None, y=None):
    if x == None:
        x = [np.amin(LEAF.Vertex[:,0]), np.amax(LEAF.Vertex[:,0])]
    if y == None:
        y = [np.amin(LEAF.Vertex[:,1]), np.amax(LEAF.Vertex[:,1])]

    keep = np.where(((LEAF.Vertex[:, 0] >= x[0]) & (LEAF.Vertex[:, 0] <= x[1]))
        & ((LEAF.Vertex[:, 1] >= y[0]) & (LEAF.Vertex[:, 1] <= y[1])))[0]

    aLEAF = LFCLSS.Topology(LEAF.comment+'_cropped', 'data')
    aLEAF.Bond = LEAF.Bond[np.all(np.isin(LEAF.Bond, keep), axis=1), :]

    x = 0
    for i in keep:
        aLEAF.Bond[aLEAF.Bond == i] = x
        x += 1

    aLEAF.Vertex = LEAF.Vertex[keep,:]
    aLEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    C0 = C0[np.all(np.isin(LEAF.Bond,keep), axis=1)]
    lengths = lengths[np.all(np.isin(LEAF.Bond,keep), axis=1)]

    return aLEAF, C0, lengths

def rotate_LEAF(LEAF, angle=-1*np.pi/2):
    R = np.array([[np.cos(angle), -1*np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])
    LEAF.Vertex = (R.dot(LEAF.Vertex.T)).T
    return LEAF

def remove_nodes(LEAF, LSHP, C0, lengths, nodes):
    LEAF.Vertex = np.delete(LEAF.Vertex, nodes, axis=0)
    rembonds = np.where(np.any(np.isin(LEAF.Bond, nodes), axis=1))[0]
    LEAF.Bond = np.delete(LEAF.Bond, rembonds, axis=0)
    C0 = np.delete(C0, rembonds)
    lengths = np.delete(lengths, rembonds)

    un = np.delete(np.arange(np.amax(LEAF.Bond)+1), nodes)
    conv = np.zeros(np.amax(LEAF.Bond) + 1)
    conv[un] = np.arange(len(un))
    LEAF.Bond[:, 0] = conv[LEAF.Bond[:, 0]]
    LEAF.Bond[:, 1] = conv[LEAF.Bond[:, 1]]

    hull = ConvexHull(LEAF.Vertex)
    LSHP.polyedge = LEAF.Vertex[hull.vertices, :]
    return LEAF, LSHP, C0, lengths

def remove_edges(LEAF, LSHP, C0, lengths, edges):
    def remove_edge(LEAF, LSHP, C0, lengths, edge):
        rembonds = np.where(np.all(np.isin(LEAF.Bond, edge), axis=1))[0]
        LEAF.Bond = np.delete(LEAF.Bond, rembonds, axis=0)
        C0 = np.delete(C0, rembonds)
        lengths = np.delete(lengths, rembonds)
        return LEAF, LSHP, C0, lengths

    for e in edges:
        LEAF, LSHP, C0, lengths = remove_edge(LEAF, LSHP, C0, lengths, e)
    return LEAF, LSHP, C0, lengths

def read_gbd(filename='shp/gbmd/gangesNetmod.shp', plot=False, sourceinds=[0],
        sinks=None, crop=False, x=[1e5, 2.5e5], y=[0, 1.5e5]):
    import geopandas as gpd
    shapefile = gpd.read_file(filename)

    if filename == 'shp/gbmd/gangesNetmod.shp':
        sinks = [8699, 8749, 8793, 8794, 8798, 8797, 8791, 8792, 8784, 8760,
                 8653, 8591, 8538, 8470, 8160, 8000, 7865, 7963, 8234, 8235,
                 8294, 8046, 8089, 7741, 7485,]

    if plot:
        shapefile.plot()
        plt.show()

    LEAF = LFCLSS.Topology('gbmd', 'data')

    LEAF.Bond = np.array([shapefile['FROM_NODE'],shapefile['TO_NODE']]).T - 1

    # Some indices are skipped over in shapefile, so this corrects for that
    un = np.sort(np.unique(LEAF.Bond)) # un = unique
    conv = np.zeros(np.amax(LEAF.Bond)+1) #conv = conversion array
    conv[un] = np.arange(len(un))
    LEAF.Bond[:,0] = conv[LEAF.Bond[:,0]]
    LEAF.Bond[:,1] = conv[LEAF.Bond[:,1]]

    LEAF.Vertex = np.zeros((np.amax(LEAF.Bond)+1,2))
    lengths = np.zeros(LEAF.Bond.shape[0])
    for i in range(len(shapefile['geometry'])):
        lengths[i] = shapefile['geometry'][i].length
        c = np.array(shapefile['geometry'][i].coords)
        p = LEAF.Bond[i,:]
        try:
            LEAF.Vertex[p,:] = c[[0,-1],:]
        except: print(p)

    # Make (0,0) at the bottom left
    LEAF.Vertex -= np.amin(LEAF.Vertex,axis=0)

    C0 = shapefile['Width']**3 #* 9.81/(100**2*0.004*shapefile['SHAPE_Leng'])
    C0 /= np.amax(C0[np.where(np.any(LEAF.Bond == sourceinds[0], axis=1))[0]])

    if crop:
        LEAF, C0, lengths = crop_LEAF(LEAF, C0, lengths, x=x, y=y)
        sinks = [0]

    LEAF.RidgeLengths = np.ones(LEAF.Bond.shape[0])
    LEAF.CellAreas = np.ones(LEAF.Bond.shape[0])

    LSHP = LFCLSS.LeafShape('from data', 0)
    hull = ConvexHull(LEAF.Vertex)
    LSHP.polyedge = LEAF.Vertex[hull.vertices,:]

    delta = DeltaNetwork(LEAF, LSHP, 0, 0, C0=C0,
                         fluctuations='river', basins='generate',
                         sourceinds=sourceinds, sinkinds=sinks)

    delta.lengths = lengths
    return delta

def read_deltas(delta='Colville', file='shp/DeltasNets_CIMPY', crop=False):
    from scipy.io import loadmat
    a = loadmat(file)

    LEAF = LFCLSS.Topology(delta, 'data')

    A = a[delta]['AdjW'][0][0]

    print(A)
    inds = np.nonzero(A)
    LEAF.Bond = np.array(inds).T
    LEAF.Bond = LEAF.Bond[:,::-1]
    C0 = A[inds]

    x = a[delta]['nx'][0][0][:,0]
    y = a[delta]['ny'][0][0][:,0]
    LEAF.Vertex = np.vstack((x,y)).T

    for n in range(LEAF.Vertex.shape[0]):
        if n == 0:
            continue
        prev = np.where(LEAF.Bond[:,1] == n)
        loc = np.where(LEAF.Bond[:,0] == n)
        C0[loc] *= np.amax(prev)

    C0 **= 3

    LEAF = rotate_LEAF(LEAF)
    if delta == 'Mossy':
        LEAF = rotate_LEAF(LEAF)
    elif delta == 'Parana':
        LEAF = rotate_LEAF(LEAF, angle=np.pi/8)

    #Make (0,0) at the bottom left
    LEAF.Vertex -= np.amin(LEAF.Vertex,axis=0)

    if crop:
        LEAF, C0 = crop_LEAF(LEAF, C0, x=x, y=y)
        sinks = [0]

    # correct conductivities here

    LEAF.RidgeLengths = np.ones(LEAF.Bond.shape[0])
    LEAF.CellAreas = np.ones(LEAF.Bond.shape[0])

    LSHP = LFCLSS.LeafShape('from data', 0)
    hull = ConvexHull(LEAF.Vertex)
    LSHP.polyedge = LEAF.Vertex[hull.vertices,:]

    sinks = {
        'Mossy': [58, 57, 56, 55, 54, 53, 49, 43, 42, 40, 39, 33, 32, 28,
                  27, 25, 19, 18, 13],
        'Colville': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 64, 65, 66, 68,
                     79, 80, 101, 102, 103, 104]
    }

    return DeltaNetwork(LEAF, LSHP, 0, 0, C0=C0,
        fluctuations='river', basins='generate',
        sourceinds=[0], sinkinds=sinks[delta])

def recover_json(file):
    return DeltaNetwork.load('deltadata/'+file)

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
deltas = list(T.keys())

def read_json(file='Irrawaddy', crop=False, convex=True):
    try: return recover_json(file)
    except FileNotFoundError: pass
    if file in ['Ras Isa', 'Sarawak']:
        return file

    import pandas as pd
    import json
    n = pd.DataFrame(
        json.load(open('shp/'+file+'/'+file+'_nodes.json', 'r'))['features'])
    e = pd.DataFrame(
        json.load(open('shp/'+file+'/'+file+'_links.json', 'r'))['features'])

    LEAF = LFCLSS.Topology(file, 'data')

    start = [eval(x['conn'])[0] for x in e['properties']]
    fin = [eval(x['conn'])[1] for x in e['properties']]
    LEAF.Bond = np.array([start, fin]).T

    un = np.array([x['id'] for x in n['properties']])
    conv = np.zeros(np.amax(LEAF.Bond) + 1)
    conv[un] = np.arange(len(un))
    LEAF.Bond[:, 0] = conv[LEAF.Bond[:, 0]]
    LEAF.Bond[:, 1] = conv[LEAF.Bond[:, 1]]

    if file == 'Mississippi':
        LEAF.Bond = LEAF.Bond[1:, :]

    x = [x['coordinates'][0] for x in n['geometry']]
    y = [x['coordinates'][1] for x in n['geometry']]

    LEAF.Vertex = np.array([x, y]).T
    LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)
    LEAF.Vertex *= 1e-3

    try: w = np.array([eval(x['wid']) for x in e['properties']])
    except KeyError: w = np.ones(LEAF.Bond.shape[0])
    # length = np.array([eval(x['len_adj']) for x in e['properties']])
    C0 = w**3 #* 9.8/0.004/1e2**2
    try: lengths = np.array([float(x['len']) for x in e['properties']])
    except:
        lengths = np.linalg.norm(LEAF.Vertex[LEAF.Bond[:,0],:] - \
            LEAF.Vertex[LEAF.Bond[:,1],:], axis=1)

    if file == 'Mississippi':
        C0 = C0[1:]
        lengths = lengths[1:]

    if file == 'Niger':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 115], y=[-1, 1e3])
        LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    if file == 'Yenisei':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 80], y=[20, 180])

    if file == 'Yukon':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 1e3], y=[36.75, 1e3])

    if file == 'Colville':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 1e3], y=[8, 1e3])

    if file == 'Apalachicola':
        LEAF, C0, lengths = crop_LEAF(LEAF, C0, lengths, y=[0,15])

    LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    sources = {'St Clair': 0,
               'Mississippi': 7,
               'Wax': 18,
               'Mossy': 31,
               'Kolyma': 17,
               'Colville': 97,
               'Apalachicola': 0,
               'Mackenzie': 593,
               'Orinoco': 196,
               'Yenisei': 336,
               'Lena': 1898,
               'Yukon': 268,
               'Betsiboka': 56,
               'Irrawaddy': 288,
               'GBM': 373,
               'Rajang': 49,
               'Niger': 124,
               'Sarawak': 0,
               'Ras Isa': 0,
               'Barnstable': 37}

    #sinks = {x: [1] for x in deltas}
    with open('shp/sinks.p','rb') as f:
        sinks = {x[0]: x[1] for x in zip(deltas, pickle.load(f))}

    for d in deltas:
        try: sinks[d].remove(sources[d])
        except ValueError: pass

    firstbondind = np.where(np.any(LEAF.Bond == sources[file], axis=1))
    cscale = np.amax(C0[firstbondind])
    if file == 'Wax':
        cscale = 1000**3
    C0 /= cscale
    lengths /= cscale**(1/3)

    LEAF.RidgeLengths = np.ones(LEAF.Bond.shape[0])
    LEAF.CellAreas = np.ones(LEAF.Bond.shape[0])

    LSHP = LFCLSS.LeafShape('from data', 0)
    if convex:
        hull = ConvexHull(LEAF.Vertex)
        LSHP.polyedge = LEAF.Vertex[hull.vertices, :]
    else:
        import alphashape
        alpha = 0.95 * alphashape.optimizealpha(LEAF.Vertex)
        hull = alphashape.alphashape(LEAF.Vertex, alpha)
        LSHP.polyedge = np.array(hull.exterior.coords.xy).T

    if sinks is None:
        sinks = [LEAF.Vertex.shape[0] - 1]

    if file == 'St Clair':
        edges = [(1,65)]
        LEAF, LSHP, C0, lengths = \
            remove_edges(LEAF, LSHP, C0, lengths, edges)

    if file == 'Barnstable':
        nodes = [6, 4, 2, 1, 0, 5, 8, 10, 7, 11, 13, 15, 12, 16, 17, 14, 26,
            31, ]
        edges = [(31,35), (35, 41), (41,48), (48, 55), ]
        LEAF, LSHP, C0, lengths = \
            remove_edges(LEAF, LSHP, C0, lengths, edges)
        LEAF, LSHP, C0, lengths = \
            remove_nodes(LEAF, LSHP, C0, lengths, nodes)

    delta = DeltaNetwork(LEAF, LSHP, 0, 0, C0=C0,
        fluctuations='river', basins='generate',
        sourceinds=[sources[file]], sinkinds=sinks[file])

    delta.bond_lens = lengths

    return delta

def newjson(file):
    if file in ['Ras Isa', 'Sarawak']:
        return file

    import pandas as pd
    import json
    n = pd.DataFrame(
        json.load(open('shp/'+file+'/'+file+'_nodes.json', 'r'))['features'])
    e = pd.DataFrame(
        json.load(open('shp/'+file+'/'+file+'_links.json', 'r'))['features'])

    LEAF = LFCLSS.Topology(file, 'data')

    start = [eval(x['conn'])[0] for x in e['properties']]
    fin = [eval(x['conn'])[1] for x in e['properties']]
    simpleBond = np.array([start, fin]).T

    un = np.array([x['id'] for x in n['properties']])
    conv = np.zeros(np.amax(simpleBond) + 1)
    conv[un] = np.arange(len(un))
    simpleBond[:, 0] = conv[simpleBond[:, 0]]
    simpleBond[:, 1] = conv[simpleBond[:, 1]]

    x = [x['coordinates'][0] for x in n['geometry']]
    y = [x['coordinates'][1] for x in n['geometry']]

    LEAF.Vertex = np.array([x, y]).T

    try: w = np.array([eval(x['wid']) for x in e['properties']])
    except KeyError: w = np.ones(LEAF.Bond.shape[0])
    simpleC0 = w**3 #* 9.8/0.004/1e2**2

    skip = 10
    if file == 'Barnstable':
        rembonds = [(31,35), (35, 41), (41,48), (48, 55), ]
        remnodes = [6, 4, 2, 1, 0, 5, 8, 10, 7, 11, 13, 15, 12, 16, 17, 14, 26,
            31, ]
    else:
        rembonds = []
        remnodes = []
    startedlist = False
    for i in np.arange(simpleBond.shape[0]):
        start, end = simpleBond[i,:]
        if (start, end) in rembonds:
            continue
        if start in remnodes or end in remnodes:
            continue

        nextnewnode = LEAF.Vertex.shape[0]
        xy = np.array(e['geometry'][i]['coordinates'])
        if xy.shape[0] == 2:
            LEAF.Bond = np.append(LEAF.Bond, [[start, end]], axis=0)
            C0 = np.append(C0, [simpleC0[i]])
            continue
        if xy.shape[0] >= 3*skip:
            xy = xy[::skip,:]

        newbonds = np.zeros((xy.shape[0]-2, 2), dtype='int')

        newbonds[0,0] = start
        newbonds[-1,1] = end

        newbonds[1:,0] = np.arange(nextnewnode, nextnewnode+newbonds.shape[0]-1)
        newbonds[:-1,1]= np.arange(nextnewnode, nextnewnode+newbonds.shape[0]-1)

        if not startedlist:
            LEAF.Bond = newbonds
            C0 = np.repeat(simpleC0[i], newbonds.shape[0])
            startedlist = True
        else:
            LEAF.Bond = np.append(LEAF.Bond, newbonds, axis=0)
            C0 = np.append(C0, np.repeat(simpleC0[i], newbonds.shape[0]))
        LEAF.Vertex = np.append(LEAF.Vertex, xy[1:-1], axis=0)

    LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)
    LEAF.Vertex *= 1e-3

    lengths = np.linalg.norm(LEAF.Vertex[LEAF.Bond[:,0],:] - \
        LEAF.Vertex[LEAF.Bond[:,1],:], axis=1)

    if file == 'Mississippi':
        LEAF.Bond = LEAF.Bond[1:, :]

    if file == 'Mississippi':
        C0 = C0[1:]
        lengths = lengths[1:]

    if file == 'Niger':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 115], y=[-1, 1e3])
        LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    if file == 'Yenisei':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 80], y=[20, 180])
        LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    if file == 'Yukon':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 1e3], y=[36.75, 1e3])
        LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    if file == 'Colville':
        LEAF, C0, lengths = crop_LEAF(
            LEAF, C0, lengths, x=[-1, 1e3], y=[8, 1e3])
        LEAF.Vertex -= np.amin(LEAF.Vertex, axis=0)

    if file == 'Apalachicola':
        LEAF, C0, lengths = crop_LEAF(LEAF, C0, lengths, y=[0,13.5])

    sources = [1, 7, 52, 31, 17, 97, 20, 593, 196, 336, 1898, 268, 56, 289, 49, 0, 124, 37]
    sources = {x[0]: [x[1]] for x in zip(deltas,sources)}

    with open('shp/sinks.p','rb') as f:
        sinks = {x[0]: x[1] for x in zip(deltas,pickle.load(f))}
    sinks['Wax'] = [10]
    #b = sinks['Niger']
    #sinks['Barnstable'] = b
    #sinks['Niger'] = [1]

    sinks['Barnstable'] = [27]
    sources['Barnstable'] = [28]

    for d in deltas:
        try: sinks[d].remove(sources[d])
        except ValueError: pass
        except KeyError: pass

    firstbondind = np.where(np.any(LEAF.Bond == sources[file], axis=1))
    cscale = np.amax(C0[firstbondind])
    if file == 'Wax':
        cscale = 1000**3
    C0 /= cscale
    lengths /= cscale**(1/3)

    LEAF.RidgeLengths = np.ones(LEAF.Bond.shape[0])
    LEAF.CellAreas = np.ones(LEAF.Bond.shape[0])

    LSHP = LFCLSS.LeafShape('from data', 0)

    hull = ConvexHull(LEAF.Vertex)
    LSHP.polyedge = LEAF.Vertex[hull.vertices, :]

    if sinks is None:
        sinks = [LEAF.Vertex.shape[0] - 1]

    delta = DeltaNetwork(LEAF, LSHP, 0, 0, C0=C0,
        fluctuations='river', basins='generate',
        sourceinds=sources[file], sinkinds=sinks[file])

    delta.bond_lens = lengths

    return delta

def getsinks():
    sinks = []
    for k in deltas:
        print(k)
        a = read_json(k)
        if isinstance(a, str):
            sinks.append([])
            continue

        ax = plt.subplot(111)
        a.to_networkx(thr=1e-10)

        pos = nx.get_node_attributes(a.G, 'pos')
        nx.draw_networkx(a.G, pos, with_labels=False, node_size=6,
                         node_color='black', ax=ax)
        x = [d[0] for d in a.G.degree if d[1]==1]
        sinks.append(x)
        #SG = a.G.subgraph(x)
        #nx.draw_networkx(SG, pos, with_labels=True, node_size=6,
        #    node_color='black', ax=ax)
        #a.drawspecial()
        #plt.show()
        plt.clf()

    with open('shp/sinks.p','wb') as f:
        pickle.dump(sinks, f)
def getsources():
    for k in deltas:
        print(k)
        a = read_json(k)
        if isinstance(a, str):
            continue
        a.to_networkx(thr=1e-10)

        pos = nx.get_node_attributes(a.G, 'pos')
        nx.draw_networkx_labels(a.G, pos)
        x = [d[0] for d in a.G.degree if d[1]==1]
        #SG = a.G.subgraph(x)
        #nx.draw_networkx(SG, pos, with_labels=True)
        a.plot(alpha=True)
        plt.show()

if __name__ == '__main__':
    """ Example code """

    # Load and show one of the deltas
    a = read_json('Lena')
    a.plot()
    plt.show()


    # simulate a delta
    a = DeltaNetwork.make_river(1, 1, density=80, shape='triangle')
    a.simulate()
    a.plot()
    plt.show()
