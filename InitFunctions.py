import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import LEAFclass as LFCLSS
import matplotlib.pyplot as plt
import matplotlib.path as pltth


#INCLUDED FUNCTIONS
# Create_networkFO
# leafshape
# neighborsF
# refineF
# displayNetworkF

def polygon_area(coords):
    """ Return the area of a closed polygon
    """
    Xs = coords[:,0]
    Ys = coords[:,1]

    # Ignore orientation
    return 0.5*abs(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))

def replace_nan_by_avg(ar):
    # replace all nans by the average of the rest
    avg = ar[np.isfinite(ar)].mean()
    ar[np.isnan(ar)] = avg

    return ar

def Create_networkFO(leafName, density, lattice, LSHP, yplot=False, \
    angle=np.pi/3., noise=0.0, zoom_factor=1.0, shapeplot=False, stats=False,
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
        X = linspace(-1.5, 1.5, num=density)
        Y = np.zeros(density)

        LEAF.height = X[1] - X[0]

        Y2 = arange(LEAF.height, 1, LEAF.height)
        X2 = X[len(X)/3]*np.ones(len(Y2))

        Y3 = arange(-LEAF.height, -1, -LEAF.height)
        X3 = X[len(X)/3]*np.ones(len(Y3))

        maxlength = LEAF.height*1.01
        VertexM = np.zeros((density + len(Y2) + len(Y3), 2))
        VertexM[:, 0] = concatenate((X, X2, X3))
        VertexM[:, 1] = concatenate((Y, Y2, Y3))

    elif lattice == 'hjunc':
        Mnei = 2
        X = linspace(-1.5, 1.5, num=density)
        Y = np.zeros(density)

        LEAF.height = X[1] - X[0]

        Y2 = arange(LEAF.height, 1, LEAF.height)
        X2 = X[len(X)/3]*np.ones(len(Y2))

        Y3 = arange(-LEAF.height, -1, -LEAF.height)
        X3 = X[len(X)/3]*np.ones(len(Y3))

        Y4 = arange(LEAF.height, 1, LEAF.height)
        X4 = X[len(X)/3 + 4]*np.ones(len(Y4))

        Y5 = arange(-LEAF.height, -1, -LEAF.height)
        X5 = X[len(X)/3 + 4]*np.ones(len(Y5))

        maxlength = LEAF.height*1.01
        VertexM = np.zeros((density + len(Y2) + len(Y3) + len(Y4) + len(Y5), 2))
        VertexM[:, 0] = concatenate((X, X2, X3, X4, X5))
        VertexM[:, 1] = concatenate((Y, Y2, Y3, Y4, Y5))

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

        """Mnei = 6;
        nrows = density + 2;
        ncolu = density + 2;

        X = np.zeros(nrows*ncolu);
        Y = np.zeros(nrows*ncolu);

        awidth = 3./nrows; #advice: adjust the spacing according to nrows
        aheight = np.sin(angle)*awidth

        Xmax = awidth/2. * ncolu
        X[0] = -Xmax
        Xmin = -Xmax
        Ymax = aheight/2. * nrows
        Y[0] = -Ymax
        Ymin = -Ymax

        LEAF.height = aheight;

        c=0;
        for nr in range(nrows):
            if np.mod(nr,2)==0:
                for nc in range(ncolu):

                    Y[c]=Y[0]+((nr)*aheight);
                    X[c]=X[0]+((nc)*awidth);
                    c=c+1;
                ##end
            else:
                for nc in range(ncolu-1) :

                    Y[c]=Y[0]+((nr)*aheight);
                    X[c]=X[0]+((nc+1./2.)*awidth);
                    c=c+1;
                ##end
                #last value of c here reflectc total length
                X[c-nrows+1:c]=X[c-1:c-nrows:-1].copy();#X(c-nrows+2:c)=fliplr(X(c-nrows+2:c));
                Y[c-nrows+1:c]=Y[c-1:c-nrows:-1].copy();#Y(c-nrows+2:c)=fliplr(Y(c-nrows+2:c));
            ##end
        ##end

        X = X[:c]
        Y = Y[:c]

        if noise > 0.0:
            # move positions around randomly
            X += noise*awidth*(2*np.random.random(X.shape) - 1)
            Y += noise*awidth*(2*np.random.random(Y.shape) - 1)

        # maximum bond length we allow in this network
        maxlength = awidth*(1.01 + 2*noise);

        VertexM= np.zeros((c,2))
        del c
        VertexM[:,0]= X
        VertexM[:,1]= Y"""

    elif lattice == 'line':
        Mnei = 2
        X = linspace(-1.5, 1.5, num=density)
        Y = np.zeros(density)

        LEAF.height = X[1] - X[0]

        maxlength = LEAF.height*1.01
        VertexM = np.zeros((density, 2))
        VertexM[:, 0] = X
        VertexM[:, 1] = Y

    elif lattice == 'square':
        x = np.linspace(-1, 1, density)
        y = np.linspace(-1, 1, density)
        Nnodes_y = y.size

        maxlength = (x[1] - x[0])*(1.01 + 2*noise)

        x, y = [a.flatten() for a in np.meshgrid(x,y)]

        if noise > 0.0:
            # move positions around randomly
            x += noise*3.2/density*(2*np.random.random(x.shape) - 1)
            y += noise*3.2/density*(2*np.random.random(y.shape) - 1)

        VertexM = np.array([x, y]).T

    elif lattice == 'rect':
            Mnei = 4

            x = linspace(0, 2.5, density)
            y = linspace(-1.05, 1.05, 2*density)

            maxlength = (x[1] - x[0])*1.01

            X, Y = meshgrid(x, y)

            X = reshape(X, (2*density**2, 1))
            Y = reshape(Y, (2*density**2, 1))

            x = X[:,0]
            y = Y[:,0]

            VertexM = np.array([x, y]).T

    else:
        # load lattice from text file
        VertexM = loadtxt(lattice, delimiter=',')
        n_points = VertexM.shape[0]

        VertexM *= 2.42
        VertexM += np.array([1.2, 0])

        LEAF.height = max(VertexM[:,1]) - min(VertexM[:,1])

        VertexM *= zoom_factor

        maxlength = None

    #VertexM[:,0] -= min(VertexM[:,0]);
    #VertexM[:,1] -= np.mean(VertexM[:,1]);

    xyleaf = leafshape(LSHP, shapeplot);

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
            ridge_lens[i] = nan
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

## --------------------------------------------------------

def leafshape(LSHP, plot=False):

    #theta = linspace(0,np.pi, round(np.pi/0.01));

    ##width = LSHP.width ;
    ##nsharp = LSHP.nsharp; %small-->sharp leaves
    ##basewidth = LSHP.basewidth;
    ##baselength = LSHP.baselength;
    ##aspectr = LSHP.aspectr; %large->long leaves
    ##polyedge = LSHP.polyedge;
    ##q = LSHP.q;

    if len(LSHP.polyedge)==0:
        print ('does not work for python if polyedge is empty')
        ##    r = (1.-LSHP.aspectr*abs(2.*(theta-np.pi/2.)/np.pi))*exp(-(abs((mod(LSHP.q*theta,np.pi))-np.pi/2.)/LSHP.width)**nsharp  );
        ##
        ##    [x,y] = pol2cart(theta,r);
        ##    y = y+LSHP.baselength;
        ##
        ##    y = [0 y 0];
        ##    x = [LSHP.basewidth x -LSHP.basewidth];
    else:
        x = LSHP.polyedge[0,:]
        y = LSHP.polyedge[1,:]

    #end

    if plot:
        plt.plot(x,y)
        plt.show()

    xy = np.vstack((x,y))

    return xy

## --------------------------------------------------------
"""
fing neighbors from bonds, or bonds from neighbors
"""
def  neighborsF(InputM,Mnei,Np2):


    if InputM.shape[1]==2:

        BondM = InputM.copy()
        NeighM = np.zeros((Np2,Mnei))
        for jc in xrange(Np2):
            K1 = nonzero(jc == BondM[:,0]);
            K2 = nonzero(jc == BondM[:,1]);

            neigh0 = hstack((BondM[K1[0],1], BondM[K2[0],0]))
            if len(neigh0)!=Mnei:
                neigh = hstack((neigh0, -1*np.ones(Mnei-len(neigh0)) ))
            else:
                neigh = neigh0
            #end
            NeighM[jc,:] = neigh
        #end
        OutputM = NeighM.copy()
    else:
        NeighM = InputM.copy()
        BondM=[];

        for jc in xrange(len(NeighM)):
            neigh = NeighM[jc,:];
            neigh = neigh[neigh>jc];
            dBondM = hstack((tile(jc,(len(neigh),1)), neigh));
            BondM = np.vstack((BondM, dBondM))
        #end
        BondM = sort(BondM,axis=1);
        OutputM = BondM;


    return OutputM

## --------------------------------------------------------
"""
% subdivide network
%DOES NOT WORK WITH PERIODIC BC
"""
def  refineF(LEAF,vectorConductivity,gam,condcheck):

    LEAF2 = LFCLSS.Topology(LEAF.comment+'refined')
    #initial network
    BondM = LEAF.Bond;
    VertexM = LEAF.Vertex;
    NeighM = LEAF.Neigh;

    Mnei = NeighM.shape[1];
    Np2 = len(VertexM);
    LBond = len(BondM)

    if condcheck=='areole':
        print ('python not yet coded for areole')
        #Nsources sources are in the end
        ##        NSources = LEAF.NSources ;
        ##        Ksource  = Np2-(NSources-1:-1:0);
        ##        Kcon = cell(NSources,1);
        ##        K = [];
        ##        for js = 1:length(Ksource)
        ##            [Kr nc] = find(BondM==Ksource(js));
        ##            tmp = BondM(Kr,:);tmp(tmp == Ksource(js)) = [];
        ##            Kcon{js} = tmp';
        ##            K = [K; Kr];
        ##        end
        ##        LBEnd = length(K) ;
    else:
        LBEnd = 0;
        NSources = 0;
    #end

    #%%%%%%%BODY
    ##    indices = BondM[1:end-LBEnd,1)+(BondM(1:end-LBEnd,2)-1)*Np2;
    ##    condvec = LEAF.Conductivity(indices);
    condvec  = vectorConductivity[:BondM.shape[0]-LBEnd].copy()

    #each bond in body will now be split to 2
    le = BondM.shape[0]-LBEnd;
    eBond = np.zeros((LBond-LBEnd+le,2))
    eBond[:,0] = hstack((BondM[0:LBond-LBEnd,0], arange(Np2-NSources, le+Np2-NSources)))
    eBond[:,1] = hstack((BondM[0:LBond-LBEnd,1], arange(Np2-NSources, le+Np2-NSources)))
    econdvec = hstack((condvec, condvec));


    #one new point per bond
    dVertex = np.zeros((LBond-LBEnd,2))
    dVertex[:,0] = (VertexM[BondM[:LBond-LBEnd,0], 0] + VertexM[BondM[:LBond-LBEnd,1], 0] )/2.
    dVertex[:,1] = (VertexM[BondM[:LBond-LBEnd,0], 1] + VertexM[BondM[:LBond-LBEnd,1], 1] )/2.;
    VertexN =  np.vstack((VertexM[:Np2-NSources,:], dVertex));    #will add Sources later

    #find connectivity
    #dum, BondN, FacetMp, dum = MtDLN.delaunay(VertexN[:,0],VertexN[:,1])
    triang = Triangulation(VertexM[:,0], VertexM[:,1])
    BondN = triang.edges

    #sort BondN
    BondN = sort(BondN, axis=1)

    tempdv =np.sqrt( ( VertexN[BondN[:,0],0]-VertexN[BondN[:,1],0] )**2 + \
                  ( VertexN[BondN[:,0],1]-VertexN[BondN[:,1],1] )**2);

    maxlength = 1.0001* min(tempdv);

    #remove outer higher length bonds
    vecX = np.zeros(BondN.shape)
    vecX[:,0] = VertexN[BondN[:,0],0]
    vecX[:,1] = VertexN[BondN[:,1],0]
    vecY = np.zeros(BondN.shape)
    vecY[:,0] = VertexN[BondN[:,0],1]
    vecY[:,1] = VertexN[BondN[:,1],1]
    K = np.sqrt(np.diff(vecX)**2+np.diff(vecY)**2)<=maxlength;

    BondN = BondN[np.squeeze(K),:].copy();

    Np2N = VertexN.shape[0]; #actual number of nodes
    LBondsN = BondN.shape[0]
    BondN = sort(BondN, axis=1)


    #construct neighbor list
    #NeighM = neighborsF(BondM,Mnei,Np2);
    #-----------------------




    #%New conductivitu
    #% [max(eBond(:))  Np2N NSources]
    CResN = np.zeros((Np2N,Np2N));
    indicesN = zip(*eBond)

    CResN[indicesN] = econdvec
    indicesA = zip(*BondN)

    temp = -log(np.random.rand(len(BondN)));
    CResN[indicesA] = CResN[indicesA] + temp/max(temp) * scipy.stats.gnp.mean(econdvec[econdvec!=0])+0;

    CResN = CResN+CResN.T
    CResN = CResN/sum(sum(CResN**gam))**(1/gam);
    vectorConductivityNew = CResN[zip(*BondN)]

    if condcheck == 'areole':
        print('not set up fr areole')
        ##        %%%%%%%EDGE
        ##        VertexN = [VertexN; VertexM(Np2-NSources+1:Np2,:)];
        ##        CResN = [CResN np.zeros(size(CResN,1),NSources)];
        ##        CResN = [CResN; np.zeros(NSources,size(CResN,2))];
        ##
        ##        mxK = Mnei;
        ##        for js =1:NSources
        ##            BondM1 = [Kcon{js} (Np2N+js)*np.ones(length(Kcon{js}),1)];
        ##            BondN = [BondN; BondM1];
        ##
        ##            CKmC1 = max(CResN(BondM1(:,1),:),[],2);
        ##            CResN(Kcon{js}, Np2N+js) = CKmC1;
        ##            mxK = max(mxK, length(Kcon{js}));
        ##        end
        ##
        ##
        ##        % assignin('base', 'CResN', CResN);
        ##        % assignin('base', 'CKmC1', CKmC1);
        ##
        ##
        ##
        ##
        ##
        ##        CResN = CResN+CResN';
        ##        CResN = CResN/sum(sum(CResN.^gam))^(1/gam);
        ##
        ##        Np2N = length(VertexN); %actual number of nodes NEW
        ##
        ##        NeighN = neighborsF(BondN,mxK,Np2N);
        ##        %%%%%%%%%
        ##
        ##
        ##        %set structure
        ##        LEAF2.Vertex = VertexN;
        ##        LEAF2.Neigh = NeighN;
        ##        LEAF2.Bond = BondN;
        ##        LEAF2.Conductivity = CResN;
        ##        LEAF2.gamma = gam;

    else:
        NeighN = neighborsF(BondN,Mnei,Np2N);
        LEAF2.Vertex = VertexN;
        LEAF2.Neigh = NeighN;
        LEAF2.Bond = BondN;
        LEAF2.gamma = gam;


    #end


    return LEAF2, vectorConductivityNew

##------------------------------------------------------------
"""
% displayNetworkF.m VERSION V.0.0
% ELENI KATIFORI, Dec 2010

% This function plots the widths of each link

% INPUT: TOPOL: structure with the following fields
%        x: vertex coordinate list
%        fld: np.diffusion coeff of every bond

%        magn: magnification of plot
%        funchnd: function handle that rescales d for plotting
%        funchcolor
%        domiffing

% OUTPUT: plot CIRCUIT
"""

def displayNetworkF(*varargin):
    #print 'TOPOL, magnitude, funchnd, plotted field'

    TOPOL = varargin[0];
    # defaults
    magn = 1;
    funchnd = lambda x: x/max(x)
    fld = np.ones(len(TOPOL.Bond));
    funchcolor= funchnd;
    domiffing =0;
    dorotate = 0;
    Col = np.array([1.,1.,1.]);
    concentrations = None
    currents = None


    numelVarargin = len(varargin);
    if numelVarargin ==1:
        pass
    elif numelVarargin ==2:
        magn = varargin[1];
    elif numelVarargin ==3:
        magn = varargin[1];
        funchnd = varargin[2];
    elif numelVarargin ==4:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
    if numelVarargin ==5:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
    elif numelVarargin ==6:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
        domiffing =varargin[5];
    elif numelVarargin ==7:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
        domiffing =varargin[5];
        dorotate = varargin[6];
    elif numelVarargin ==8:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
        domiffing =varargin[5];
        dorotate = varargin[6];
        Col = varargin[7];
    elif numelVarargin == 9:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
        domiffing =varargin[5];
        dorotate = varargin[6];
        Col = varargin[7];
        concentrations = varargin[8]
    elif numelVarargin == 10:
        magn = varargin[1];
        funchnd = varargin[2];
        fld = varargin[3];
        funchcolor= varargin[4];
        domiffing =varargin[5];
        dorotate = varargin[6];
        Col = varargin[7];
        concentrations = varargin[8]
        currents = varargin[9]

    if not(funchnd):
        funchnd = lambda x: x/max(x)


    if not(funchcolor):
        funchcolor = funchnd;

    if len(fld)==0:
        fld = TOPOL.vectorConductivity;

    x = TOPOL.Vertex.copy();
    if dorotate:
        xtemp = x.copy()
        x[:,1] = xtemp[:,0].copy()
        x[:,0] = xtemp[:,1].copy()
        del xtemp

    Bonds = TOPOL.Bond ;

    if len(fld)!=len(Bonds):
        print ('Hm, check vConductivity versus LEAF')

    pd = funchnd(fld);
    pdCol = funchcolor(fld);

    if concentrations == None:
        for i in xrange(len(Bonds)):

            if pd[i]!=0:
                dx = hstack((x[Bonds[i,0],0] , x[Bonds[i,1],0] ))
                dy = hstack((x[Bonds[i,0],1] , x[Bonds[i,1],1] ))
                plt.plot(dx, dy,linewidth=magn*pd[i], color = tuple( Col*(1. -pdCol[i])))

            if domiffing:
                dx = hstack((x[Bonds[i,0],0] , x[Bonds[i,1],0] ))
                dy = hstack((x[Bonds[i,0],1] , x[Bonds[i,1],1] ))

                plt.plot(dx, dy, marker='.',markersize=2*magn*pd[i],color = tuple( Col*(1. -pdCol[i])))

            if currents != None:
                # Arrows indicating current direction
                if currents[Bonds[i,0], Bonds[i,1]] > 0:
                    plt.arrow(x[Bonds[i,1],0], x[Bonds[i,1],1], \
                    x[Bonds[i,0],0]-x[Bonds[i,1],0], x[Bonds[i,0],1]-x[Bonds[i,1],1])
                elif currents[Bonds[i,0], Bonds[i,1]] < 0:
                    plt.arrow(x[Bonds[i,0],0], x[Bonds[i,0],1], \
                    x[Bonds[i,1],0]-x[Bonds[i,0],0], x[Bonds[i,1],1]-x[Bonds[i,0],1])


    # Plot concentrations
    else:
        plt.scatter(x[:,0], x[:,1], c=concentrations, s=70, cmap=plt.cm.jet)
        plt.colorbar()

        for i in xrange(len(Bonds)):
            if currents != None:
                # Arrows indicating current direction
                if currents[Bonds[i,0], Bonds[i,1]] > 1e-12:
                    plt.arrow(x[Bonds[i,1],0], x[Bonds[i,1],1], \
                    (x[Bonds[i,0],0]-x[Bonds[i,1],0])/2, (x[Bonds[i,0],1]-x[Bonds[i,1],1])/2, \
                    linewidth=abs(currents[Bonds[i,0], Bonds[i,1]])**0.5, head_width=0.025)
                elif currents[Bonds[i,0], Bonds[i,1]] < -1e-12:
                    plt.arrow(x[Bonds[i,0],0], x[Bonds[i,0],1], \
                    (x[Bonds[i,1],0]-x[Bonds[i,0],0])/2, (x[Bonds[i,1],1]-x[Bonds[i,0],1])/2, \
                    linewidth=abs(currents[Bonds[i,0], Bonds[i,1]])**0.5, head_width=0.025)


    plt.axes().set_aspect('equal')
    plt.axes().set_ylim([-1.3,1.3])


    return 0
