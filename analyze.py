# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:35:17 2018

Claire Dore 2017
Adam Konkol 2019
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx
import xylem as xy
from xylem import PialNetwork as pi
from xylem import DeltaNetwork as de
from shapely.geometry import Polygon, Point
import pickle
import cmath
from scipy.stats import ks_2samp
import csv

import warnings
import time
import gc

def s(theta):
    if theta <= np.pi :
        return theta
    else:
        return theta - 2*np.pi

class AnalysisClass(object):

    def data_to_pial(self, filename, dataind, rescale=True):
        #build a AnalysisClass object from a data_file
        f = scipy.io.loadmat(filename)
        self.vertices = f['G'][0][0][0][0,0]#number of vertices in the graph
        self.edges = f['G'][0][0][1][0,0]#number of edges
        x = f['G'][0][0][2][0]
        y = f['G'][0][0][3][0]
        vertices_type = f['G'][0][0][4][0]
        self.penetrating_arterioles = np.where(vertices_type==2)[0]
        self.Adj = f['G'][0][0][5] #adjacency matrix

        #rescale data to scalebars (so x,y are in mm)
        if rescale:
            imported = []
            with open('scaling_claire.csv') as f:
                filereader = csv.reader(f, delimiter=',')
                for row in filereader:
                    imported.append(row)
            scalingdata = np.array(imported[dataind]).astype('int')
            node1 = scalingdata[3]; node2 = scalingdata[4]
            a = (x[node1],y[node1])
            b = (x[node2],y[node2])
            dist = np.sqrt( (a[0]-b[0])**2+(a[1]-b[1])**2 )
            x = x - a[0]
            y = y - a[1]
            x = x/(dist*scalingdata[1]/scalingdata[2])
            y = y/(dist*scalingdata[1]/scalingdata[2])
            x = x - min(x); y = max(y) - y

        x = x.tolist()
        y = y.tolist()
        positions = zip(x,y)
        self.pos = dict(zip(range(self.vertices),positions)) #useful to plot
        rows, cols = np.where(self.Adj.todense() == 1)
        edges = zip(rows.tolist(), cols.tolist()) # contain every pair of vertices connected with an edge :(n_1,n_2) but also (n_2,n_1)
        self.G = nx.Graph() #create a graph
        self.G.add_edges_from(edges) #build the graph by adding the edge. Each edge appear twice : (n1,n2) and (n2,n1)
        for n in self.G.nodes() : #adding the position of the nodes
            self.G.node[n]['x'] = x[n]
            self.G.node[n]['y'] = y[n]
        self.sources = None

    def simulation_to_pial(self, xylem):
        #build a AnalysisClass object from a xylem object
        self.penetrating_arterioles = xylem.sinks #list of the indexes of the pa
        self.edges = xylem.bonds #number of edges
        self.G = xylem.to_networkx() #a graph
        self.vertices = xylem.verts #number of vertices
        x_pos = []
        y_pos = []
        for n in self.G.nodes() :
            x_pos.append(self.G.node[n]['x'])
            y_pos.append(self.G.node[n]['y'])
        positions = zip(x_pos,y_pos)
        self.pos = dict(zip(self.G.nodes(),positions))
        self.sources = xylem.sources
        self.sigma_rescaled = xylem.sigma

    def plot_data(self):
        #fig = plt.figure()
        #fig, ax = plt.subplots(figsize=(10, 10))
        #nx.draw_networkx(self.G, self.pos, with_labels=False,node_size=4,node_color='blue')
        #nx.draw_networkx_nodes(self.G, self.pos, nodelist=list(self.penetrating_arterioles), node_color='y',alpha=0.8,node_size=14)
        '''x_art=[self.pos[n][0] for n in self.penetrating_arterioles]
        y_art=[self.pos[n][1] for n in self.penetrating_arterioles]
        plt.plot(x_art,y_art,marker='.',color='y',linestyle='none',alpha=0.5,markersize=8)'''
        plt.axis('off')
        for pair in self.G.edges():
            x = [self.pos[pair[0]][0], self.pos[pair[1]][0] ]
            y = [self.pos[pair[0]][1], self.pos[pair[1]][1] ]
            plt.plot(x, y, 'b', alpha=1-sum(y)/2/8000)
        for sink in self.penetrating_arterioles:
            plt.plot([self.pos[sink][0]], [self.pos[sink][1]], 'y.', markersize=10,
                     alpha=1-self.pos[sink][1]/8000)

    def plot_in_color(self,node_list,color):
        #to color the node_list of your choice in the color of your choice
        x = [self.pos[n][0] for n in node_list]
        y = [self.pos[n][1] for n in node_list]
        plt.plot(x,y,marker='.',color=color,linestyle='none')

    def remove_trees(self):
        #return a copy of the AnalysisClass object without the tree-part.
        G2 = self.G.copy()
        node_list = np.array(G2.nodes)
        PAs2 = list(self.penetrating_arterioles.copy())

        #Remove any loops (self edges)
        for i in node_list:
            if G2.has_edge(i,i):
                G2.remove_edges_from([(i,i)])

        A = nx.adjacency_matrix(G2,node_list,weight=None)
        B = A.sum(axis=1)
        nodes_to_remove = node_list[np.where(B<=1)[0]]
        pos2 = dict(self.pos)
        while len(nodes_to_remove) > 0:
            for n in nodes_to_remove:
                pos2.pop(n,None)
                if n in PAs2:
                    PAs2.remove(n)
            G2.remove_nodes_from(nodes_to_remove)
            A = nx.adjacency_matrix(G2,node_list,weight=None)
            B = A.sum(axis=1)
            #nodes_to_remove= list(np.where(B==1)[0])
            nodes_to_remove = node_list[np.where(B==1)[0]]
        new_pial = AnalysisClass()
        new_pial.G = G2
        new_pial.pos = pos2
        new_pial.penetrating_arterioles = PAs2
        new_pial.sources = None
        return (new_pial)

    def length_loop(self,cycle):
        c = [self.pos[n] for n in cycle]
        polygon = Polygon(c)
        return polygon.length

    def area_loop(self,cycle):
        c = [self.pos[n] for n in cycle]
        polygon = Polygon(c)
        return polygon.area

    def loop_not_minimal(self,cycle):
        #return False if a cycle is minimal(does not contain anything inside)
        #True if the cycle is not minimal
        c = [self.pos[n] for n in cycle]
        polygon = Polygon(c)
        for n in self.G.nodes() :
            point = Point((self.G.node[n]['x'],self.G.node[n]['y']))
            if polygon.contains(point):
                return True
        return False

    def find_left_edge(self,edge):
        '''The AnalysisClass must be tree-free first by using the function remove_trees'''
        node = edge[1]
        neighbors = list(self.G.neighbors(node))
        neighbors.remove(edge[0])

        z = complex(self.G.node[node]['x']-self.G.node[edge[0]]['x'],self.G.node[node]['y']-self.G.node[edge[0]]['y'])
        z2 = [s(cmath.phase(complex(self.G.node[n]['x']-self.G.node[node]['x'],self.G.node[n]['y']-self.G.node[node]['y'])/z)) for n in neighbors]
        i = np.argmax(z2)

        left_edge = (node,neighbors[i])
        return left_edge

    def find_left_loop(self,edge,Bonds):
        #find a minimal loop, by starting from an edge (orientated) and turning left
        cycle = []
        cycle_nodes = []

        cycle.append(edge)
        cycle_nodes.append(edge[0])
        Bonds.remove(edge)

        first_node = edge[0]
        last_node = edge[1]

        while last_node != first_node:
            cycle_nodes.append(last_node)
            edge = self.find_left_edge(edge)
            last_node = edge[1]
            cycle.append(edge)
            Bonds.remove(edge)

        return(cycle,cycle_nodes)

    def find_all_minimal_loops(self):
        '''self has to be tree-free by using remove_trees first'''
        cycles = []
        cycles_nodes = []
        Bonds = []
        for edge in self.G.edges():
            Bonds.append(edge)
            Bonds.append(tuple(reversed(edge)))
        while len(Bonds)>0 :
            first = Bonds[0]
            result = self.find_left_loop(first,Bonds)
            cycles.append(result[0])
            cycles_nodes.append(result[1])
        dico = dict()
        for i in range(len(cycles_nodes)):
            if set(cycles_nodes[i]) not in dico.values():
                dico[i] = set(cycles_nodes[i])
        cycles = [cycles[i] for i in dico.keys()]
        self.cycles_edges = cycles
        self.cycles_nodes = [cycles_nodes[i] for i in dico.keys()]
        #print(len(self.cycles_nodes))
        i = 0
        ''' because the contour of the network remains
        whereas it is not a minmal loop, we have to withdraw it'''
        while i<len(self.cycles_nodes):
            if self.loop_not_minimal(self.cycles_nodes[i]):
                self.cycles_nodes.pop(i)
                self.cycles_edges.pop(i)
            else :
                i += 1

    def count_pa_per_loop(self):
        '''returns a list containing the number of penetrating arteriol on each
        loop. meaning either it is a node of the cycle either it is attached
        to a tree inside a cycle. If a pa belongs to n loops, it counts for
        1/n for each loop.'''
        cycles = self.cycles_nodes
        stats = np.zeros(len(cycles))

        polygons = [Polygon([self.pos[n] for n in cycle]) for cycle in
                    self.cycles_nodes]
        for pa in self.penetrating_arterioles:
            l = 0 #number of loops containing pa
            m = [] #indices of loops containing pa
            A = None #is pa inside a loop (True) or part of the loop (false)?
            for k in range(len(cycles)):
                if A != False :
                    point = Point((self.G.node[pa]['x'],self.G.node[pa]['y']))
                    polygon = polygons[k]
                    if polygon.contains(point):
                         A = True
                         l = 1
                         m = [k]
                         break
                if A != True :
                    if pa in cycles[k]:
                        l += 1
                        m.append(k)
                        A = False
            for p in m:
                stats[p] += 1/l
        return stats

    def compute_tree_length_per_loop(self):
        G2 = self.G.copy()
        node_list = np.array(G2.nodes)

        #Remove any loops (self edges)
        for i in node_list:
            if G2.has_edge(i,i):
                G2.remove_edges_from([(i,i)])

        A = nx.adjacency_matrix(G2,node_list,weight=None)
        B = A.sum(axis=1)
        #Position in adj mx is not necessarily node label in graph, take indices of node_list where degree==1 in A
        extremities = node_list[np.where(B==1)[0]]

        num_cycles = len(self.cycles_nodes)

        L = [[] for i in range(num_cycles)]

        #first we find the loop in which the extremities are located
        #we know the rest of the tree will be located in the same loop

        polygons = []
        for i in range(num_cycles):
            polygons.append(Polygon([self.pos[n] for n in self.cycles_nodes[i]]))

        S = False

        for i in range(num_cycles):
            poly = polygons[i]
            for n in extremities:
                point = Point((self.G.node[n]['x'], self.G.node[n]['y']))
                if poly.contains(point):
                    S = True
                    L[i].append(n)

        '''rainbow = plt.get_cmap('rainbow')
        nx.draw_networkx(G2, self.pos, node_size=5, with_labels=False,font_size=5,node_color='r')
        nx.draw_networkx_nodes(G2, self.pos, nodelist=[1140], node_color=rainbow(0),node_size=100)
        nx.draw_networkx_nodes(G2, self.pos, nodelist=L[14], node_color=rainbow(0.5),node_size=50)

        x,y = polygons[14].exterior.xy
        plt.plot(x,y,)
        plt.show()'''


        '''for n in extremities:
            for i in range(num_cycles) :
                poly = polygons[i]
                point = Point((self.G.node[n]['x'],self.G.node[n]['y']))
                if poly.contains(point):
                    S=True #means there is at least one cycle with a tree inside
                    if n not in L[i]:
                        L[i].append(n)

                    break '''

        Length = np.zeros(num_cycles)

        while S:
            #L[k] contains the nodes with one neighbor inside cycle k
            L_next=[[] for i in range(num_cycles)]
            used_nodes = []

            for i in range(num_cycles):
                for k in L[i]:
                    k_next = next(G2.neighbors(k))

                    #Calculate length of edge
                    z = complex(self.G.node[k]['x'] - self.G.node[k_next]['x'],
                                self.G.node[k]['y'] - self.G.node[k_next]['y'])
                    dist = abs(z)
                    Length[i] += dist

                    #Handle any trees that stretch across loops by ignoring any future attempts to remove k_next (see MDA303)
                    if k_next not in used_nodes:
                        used_nodes.append(k_next)
                        L_next[i].append(k_next)
                        G2.remove_node(k)
                    else: pass


                L[i] = []

            S = False
            reused_nodes_bool = False

            node_list = np.array(G2.nodes)

            A = nx.adjacency_matrix(G2,node_list,weight=None)
            B = A.sum(axis=1)

            extremities = node_list[np.where(B==1)[0]]

            for i in range(num_cycles):
                for k_next in L_next[i]:
                    if k_next in extremities:
                        if k_next not in L[i]:
                            L[i].append(k_next)
                        S = True

            '''for i in range(num_cycles):
                for k_next in L_next[i]:
                    if B[k_next] == 1:
                        L[i].append(k_next)
                        S=True'''

        return(Length)

    def compute_tree_ratio_per_loop(self):
        #tree_ratio of a loop = legnth of the trees inside this loop/perimeter of this loop
        #return ratio =list of the tree ratios
        Ratio = self.compute_tree_length_per_loop()
        for i in range(len(self.cycles_nodes)):
            perimeter = self.length_loop(self.cycles_nodes[i])
            Ratio[i] /= perimeter
        return Ratio

    def compute_numPAs_to_loop_area_per_loop(self):
        Ratio = self.count_pa_per_loop()
        for i in range(len(self.cycles_nodes)):
            area = self.area_loop(self.cycles_nodes[i])
            Ratio[i] /= area
        return Ratio

    def compute_tree_length_to_loop_area_per_loop(self):
        Ratio = self.compute_tree_length_per_loop()
        for i in range(len(self.cycles_nodes)):
            area = self.area_loop(self.cycles_nodes[i])
            Ratio[i] /= area
        return Ratio

    def loop_perimeters(self):
        perimeters = []
        for i in range(len(self.cycles_nodes)):
            p = self.length_loop(self.cycles_nodes[i])
            perimeters.append(p)
        return perimeters

    def loop_areas(self):
        areas = []
        for i in range(len(self.cycles_nodes)):
            a = self.area_loop(self.cycles_nodes[i])
            areas.append(a)
        return areas

    def find_pa_inside_loop(self,cycle):
        #does it include the pas on the loop ? in datas they are not on the loop; but in xylem they are
        c=[self.pos[n] for n in cycle]
        polygon=Polygon(c)
        L=[]
        for pa in self.penetrating_arterioles :
            point = Point(self.pos[pa])
            if polygon.contains(point):
                L.append(pa)
        return (L)

################################################################################

def meshplotter_old(sigma_list, c_list, data, title='', cbarlabel='',
        logbool=False, graphplot=False, save_folder=''):
    sigmas = np.array(sigma_list)
    sigma_avgs = (sigmas[:-1] + sigmas[1:])/2
    sigma_bounds = [0, *sigma_avgs, sigma_list[-1]+1]

    cs = np.array(c_list)
    c_avgs = (cs[:-1] + cs[1:])/2
    c_bounds = [0, *c_avgs, c_list[-1]+1]

    xbounds, ybounds = np.meshgrid(sigma_bounds, c_bounds)

    if graphplot:
        fig = plt.figure(figsize=(20,10))
        grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
    else:
        fig = plt.figure(figsize=(10,10))
        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)


    main_ax = fig.add_subplot(grid[:,:2])

    if logbool:
        mesh = plt.pcolormesh(xbounds,ybounds,np.log10(data))
    else:
        mesh = plt.pcolormesh(xbounds,ybounds,data)
    plt.yscale('log')
    plt.title(title, fontsize=20)
    plt.xlim([0,sigma_list[-1]])
    plt.ylim([min(c_list)-1e-2, max(c_list)])

    plt.xlabel('$\sigma$', fontsize=20)
    plt.ylabel('c', fontsize=20)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    if cbarlabel == '':
        cbarlabel = title
    if logbool:
        cbar.set_label('log_10 of ' + cbarlabel, rotation=270)
    else:
        cbar.set_label(cbarlabel, rotation=270)

    x,y = np.meshgrid(sigma_list,c_list)
    plt.plot(x, y, 'k.', markersize=2)
    if graphplot:
        pairs = np.array([[0,-2],[-2,-2],[0,0],[-2,0]])
        for pair in zip([[0,2],[0,3],[1,2],[1,3]],pairs):
            ax = fig.add_subplot(grid[pair[0][0], pair[0][1]])
            ax.set_axis_off()
            try:
                picklefilename = save_folder + "/pial_c%0.2f_w%0.2f%d.obj" % \
                                   (c_list[pair[1][1]],sigma_list[pair[1][0]], 1)
                with open(picklefilename, 'rb') as f:
                    netw = pickle.load(f)
                netw.plot_data()
            except: pass

            plt.title('$\sigma = %0.2f$, $c = %0.2f$' %
                      (sigma_list[pair[1][0]],c_list[pair[1][1]]))

    return mesh

def meshplotter(sigmas, cs, data, logbool=False, graphplot=False,
        savefolder='', vmin=None, vmax=None,):
    sigma_avgs = (sigmas[:-1] + sigmas[1:])/2
    sigma_bounds = [0, *sigma_avgs, sigmas[-1]+1]

    c_avgs = (cs[:-1] + cs[1:])/2
    c_bounds = [0, *c_avgs, cs[-1]+10]

    xbounds, ybounds = np.meshgrid(sigma_bounds, c_bounds)

    if graphplot:
        fig = plt.figure(figsize=(20,10))
        grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
    else:
        fig = plt.figure(figsize=(10,10))
        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)

    plt.axis('off')
    if graphplot:
        pairs = np.array([[1,-4],[-2,-4],[1,1],[-2,1]])

        for pair in zip([[0,0],[0,1],[1,0],[1,1]],pairs):
            ax = fig.add_subplot(grid[pair[0][0], pair[0][1]])
            ax.set_axis_off()
            pname = "%s/s%0.2f_c%0.2f.obj" % (savefolder,
                sigmas[pair[1][0]], cs[pair[1][1]])
            try:
                with open(pname, 'rb') as f:
                    a = pickle.load(f)
                a.plot()
            except FileNotFoundError:
                a = make_pial(sigmas[pair[1][0]], cs[pair[1][1]], n_sources=10)
                with open(pname, 'wb') as f:
                    pickle.dump(a, f)
                a.plot()

            plt.title('$\sigma = %0.2f$, $c = %0.2f$' %
                      (sigmas[pair[1][0]], cs[pair[1][1]]))

    main_ax = fig.add_subplot(grid[:,-2:])
    plt.axis('on')

    if graphplot:
        plt.plot(sigmas[pairs[:,0]], cs[pairs[:,1]], 'rx', ms=13)

    if logbool:
        mesh = plt.pcolormesh(xbounds,ybounds,np.log10(data),
                              vmin=vmin, vmax=vmax)
    else:
        mesh = plt.pcolormesh(xbounds,ybounds,data, vmin=vmin, vmax=vmax)
    plt.yscale('log')
    #plt.title(title, fontsize=20)
    plt.xlim([0,sigmas[-1]])
    plt.ylim([min(cs)-1e-2, max(cs)])

    plt.xlabel('$\sigma$', fontsize=20)
    plt.ylabel('c', fontsize=20)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    if logbool:
        cbar.set_label('$\log_10$')
    '''# For labeling the colorbar, with optional argument 'title'
    if cbarlabel == '':
        cbarlabel = title
    if logbool:
        cbar.set_label('log_10 of ' + cbarlabel, rotation=270)
    else:
        cbar.set_label(cbarlabel, rotation=270)'''

    # enable to plot dots at each tested point
    #x,y = np.meshgrid(sigmas,cs)
    #plt.plot(x, y, 'k.', markersize=1)

    return mesh

def meshplotter_inv(sigmas, cs, data, logbool=False, graphplot=False,
        savefolder='', vmin=None, vmax=None,):
    sigma_avgs = (sigmas[:-1] + sigmas[1:])/2
    sigma_bounds = [0, *sigma_avgs, sigmas[-1]+1]

    cinvs = 1/cs[::-1]
    c_avgs = (cinvs[:-1] + cinvs[1:])/2
    c_bounds = [0, *c_avgs, np.amax(cinvs)+20]

    xbounds, ybounds = np.meshgrid(sigma_bounds, c_bounds)

    if graphplot:
        fig = plt.figure(figsize=(20,10))
        grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
    else:
        fig = plt.figure(figsize=(10,10))
        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)

    plt.axis('off')
    if graphplot:
        pairs = np.array([[1,1],[-2,1],[1,-4],[-2,-4]])

        for pair in zip([[0,0],[0,1],[1,0],[1,1]],pairs):
            ax = fig.add_subplot(grid[pair[0][0], pair[0][1]])
            ax.set_axis_off()
            pname = "%s/s%0.2f_c%0.2f.obj" % (savefolder,
                sigmas[pair[1][0]], cs[pair[1][1]])
            try:
                with open(pname, 'rb') as f:
                    a = pickle.load(f)
                a.plot()
            except:
                a = pi.make(sigmas[pair[1][0]], cs[pair[1][1]], n_sources=10)
                a.simulate()
                with open(pname, 'wb') as f:
                    pickle.dump(a, f)
                a.plot()

            plt.title('$\sigma = %0.2f$, $c = %0.2f$' %
                      (sigmas[pair[1][0]], 1/cs[pair[1][1]]))

    main_ax = fig.add_subplot(grid[:,-2:])
    plt.axis('on')

    if graphplot:
        plt.plot(sigmas[pairs[:,0]], 1/cs[pairs[:,1]], 'rx', ms=13)

    if logbool:
        mesh = plt.pcolormesh(xbounds,ybounds,np.log10(data[::-1,:]),
                              vmin=vmin, vmax=vmax)
    else:
        mesh = plt.pcolormesh(xbounds,ybounds,data[::-1,:], vmin=vmin, vmax=vmax)
    plt.yscale('log')
    #plt.title(title, fontsize=20)
    plt.xlim([0,sigmas[-1]+0.15])
    plt.ylim([min(cinvs)+1e-2, max(cinvs)+10])

    plt.xlabel('Fluctuation width scale $\sigma$', fontsize=20)
    plt.ylabel('Relative fluctuation strength', fontsize=20)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    if logbool:
        cbar.set_label('$\log_{10}$(EMD of tree length/loop perimeter)', fontsize=18)
    '''# For labeling the colorbar, with optional argument 'title'
    if cbarlabel == '':
        cbarlabel = title
    if logbool:
        cbar.set_label('log_10 of ' + cbarlabel, rotation=270)
    else:
        cbar.set_label(cbarlabel, rotation=270)'''

    # enable to plot dots at each tested point
    #x,y = np.meshgrid(sigmas,cs)
    #plt.plot(x, y, 'k.', markersize=1)

    return mesh

def rolling_average(x,y, n=80):
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    i = np.argsort(x)
    x = x[i]
    y = y[i]
    return moving_average(x, n=n), moving_average(y, n=n)

################################################################################
#Old methods using the AnalysisClass analysis class

def analyze(xylem, n, save_folder):
    #n is just an index for the xylem object
    #transform xylem into a pial object, save the pial object
    #display the network and save the fig
    #return and save 2 lists w and h : w[i] tree ratio of cycle i,
    #h[i] number of pa of cycle i
    pial = AnalysisClass()
    pial.simulation_to_pial(xylem)

    new_pial = pial.remove_trees()
    new_pial.find_all_minimal_loops()
    pial.cycles_nodes = new_pial.cycles_nodes

    h = pial.count_pa_per_loop()
    w = pial.compute_tree_ratio_per_loop()
    x = pial.compute_numPAs_to_loop_area_per_loop()
    p = pial.compute_tree_length_to_loop_area_per_loop()

    np.savez(save_folder + '/c%0.2f_w%0.2f_%d' % (xylem.cst, xylem.sigma_ratio, n), h=h,w=w,x=x,p=p)

    '''fig, ax = plt.subplots(figsize=(10, 10))
    pial.plot()
    plt.savefig('network.png')'''

    #to plot and save the histogram
    '''fig2, ax = plt.subplots(figsize=(10, 10))
    h1=plt.hist(h, bins=np.arange(17), density=True, facecolor='dodgerblue')
    plt.xlabel('number of penetrating arterioles per loop')
    plt.ylabel('density')
    plt.title(r'Histogram of number of pa per loop: $\mu=100$, $\sigma=15$')
    plt.savefig('h1.png')'''

    #to plot and save the histogram
    '''fig3, ax = plt.subplots(figsize=(10, 10))
    h2=plt.hist(w, bins=20,facecolor='dodgerblue')
    plt.xlabel('tree ratio per loop')
    plt.ylabel('density')
    plt.title(r'Histogram tree ratio per loop: $\mu=100$, $\sigma=15$')
    plt.savefig('h2.png')
    print(h,w)'''

    return (h,w,x,p)

def analyze_data(filepath, dataind):
    #take a data file ( .mat) and does the statistics
    pial = AnalysisClass()
    pial.data_to_pial(filepath, dataind)

    '''filehandler = open("pial_c"+str(xylem.cst)+"_w"+str(xylem.sigma_ratio)+
                           str(n)+".obj","wb")
    pickle.dump(pial,filehandler)
    filehandler.close()'''

    '''fig, ax = plt.subplots(figsize=(10, 10))
    pial.plot()
    plt.savefig(filepath[:-4]+'.png')'''

    new_pial=pial.remove_trees()
    new_pial.find_all_minimal_loops()
    pial.cycles_nodes=new_pial.cycles_nodes
    h = pial.count_pa_per_loop()

    #np.save('pa_per_loop'.npy',h)
    '''fig2, ax = plt.subplots(figsize=(10, 10))
    plt.hist(h, bins=np.arange(17), density=True, facecolor='dodgerblue')
    plt.xlabel('number of penetrating arterioles per loop')
    plt.ylabel('density')
    plt.title(r'Histogram of number of pa per loop')
    plt.savefig(file+'histo_pa.png')'''

    w = pial.compute_tree_ratio_per_loop()

    #np.save('w_c'+str(xylem.cst)+'_w'+str(xylem.sigma_ratio)+str(n)+'.npy',h)
    '''fig3, ax = plt.subplots(figsize=(10, 10))
    plt.hist(w, bins=20,facecolor='dodgerblue')
    plt.xlabel('tree ratio per loop')
    plt.ylabel('density')
    plt.title(r'Histogram tree ratio per loop')
    plt.savefig(file+'histo_tree_ratio.png')'''

    x = pial.compute_numPAs_to_loop_area_per_loop()

    p = pial.compute_tree_length_to_loop_area_per_loop()

    return(h,w,x,p)

def analyze_several_data_files(L, plot=False):
    H_pa = np.array([])
    W_tree = np.array([])
    X_tree = np.array([])
    P_tree = np.array([])

    i = 1
    for file in L:
        print(file)
        res = analyze_data(file, i)
        H_pa = np.append(H_pa, res[0])
        W_tree = np.append(W_tree, res[1])
        X_tree = np.append(X_tree, res[2])
        P_tree = np.append(P_tree, res[3])
        i += 1
    #np.save('data_control_pa_per_loop.npy',H_pa)
    #np.save('data_control_tree_ratio_per_loop.npy',W_tree)

    print("Avg number of PAs per loop:",np.mean(H_pa),
          "\nAvg ratio length of trees in loop to loop diameter",np.mean(W_tree),
          "\nAvg ratio length of trees in loop to loop area",np.mean(P_tree))

    if plot:
        #PLOT HISTO PA PER LOOP
        fig3, ax = plt.subplots(figsize=(10, 10))
        plt.hist(H_pa, bins=20, facecolor='dodgerblue')
        plt.xlabel('Number of PAs per loop')
        plt.ylabel('Frequency')
        plt.xlim([0,75])
        plt.title(r'Distribution of the number of PAs per loop')
        plt.text(20, 140, 'Average number of PAs in a loop \n'+'$<X_{{pa}}>=${:.1f} ± {:.1f}'.format(np.mean(H_pa),np.std(H_pa)),
            bbox={'facecolor':'lightblue', 'alpha':0.5, 'pad':10})
        plt.savefig('raw_pa_counts.png')

        #PLOT HISTO TREE RATIO PER LOOP
        fig3, ax = plt.subplots(figsize=(10, 10))
        plt.hist(W_tree, bins=25, facecolor='darkred')
        plt.xlabel('Ratio of tree length inside loop to loop perimeter [unitless]')
        plt.ylabel('Frequency')
        plt.title(r'Tree length to loop perimeter ratios')
        plt.text(0.5, 50, 'Average tree ratio per loop : \n'+'$<F_{{tree}}>=${:.3f} ± {:.2f}\nTotal number of loops: {}'.format(np.mean(W_tree),np.std(W_tree),len(W_tree)),
            bbox={'facecolor':'lightsalmon', 'alpha':0.5, 'pad':10})
        plt.savefig('raw_perimeter_ratios.png')

        #PLOT PAs per area
        pruned = X_tree[np.nonzero(X_tree<80)]#[np.nonzero(X_tree<0.001)]

        fig3, ax = plt.subplots(figsize=(10, 10))
        plt.hist(pruned, bins=20, facecolor='g')
        plt.xlabel('Density of PAs in loop [1/mm^2]')
        plt.ylabel('Frequency')
        plt.title(r'PA density per loop (PAs/loop area)')
        plt.text(1, 40, 'Average PA count to area ratio per loop : \n'+'$<F_{{tree}}>=${:.2E} ± {:.2E}\nTotal number of loops: {}'.format(np.mean(X_tree),np.std(X_tree),len(X_tree)),
            bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':10})
        plt.savefig('raw_PA_densities.png')

        fig3, ax = plt.subplots(figsize=(10, 10))
        plt.hist(P_tree, bins=30, facecolor='goldenrod')
        plt.xlabel('Ratio of tree length inside loop to loop area [1/mm]')
        plt.ylabel('Frequency')
        plt.title(r'Tree length to loop area ratios')
        plt.text(2.5, 30, 'Average tree to area ratio per loop : \n'+'$<F_{{tree}}>=${:.2E} ± {:.2E}\nTotal number of loops: {}'.format(np.mean(pruned),np.std(pruned),len(pruned)),
            bbox={'facecolor':'wheat', 'alpha':0.5, 'pad':10})
        plt.savefig('raw_area_ratios.png')

    return H_pa, W_tree, X_tree, P_tree

def study(save_folder, sigma_list, c_list, start_n, end_n, min_sigma=0,
        c_bounds=[0, 1000]):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for n in range(start_n, end_n):
            for sigma in sigma_list:
                for c in c_list:
                    if sigma >= min_sigma and c >= c_bounds[0] and c <= c_bounds[1]:
                        xylem = make_xylem(sigma,c)

                        pial = AnalysisClass()
                        pial.simulation_to_pial(xylem)

                        filehandler = open(save_folder + "/pial_c%0.2f_w%0.2f%d.obj" % \
                                           (xylem.cst, xylem.sigma_ratio,n),"wb")
                        pickle.dump(pial,filehandler)
                        filehandler.close()

                        try:
                            analyze(xylem, n, save_folder)
                        except:
                            pass

                        gc.collect()

def batch_analysis(save_folder, sigma_list, c_list, max_n, logbool=False,
        plot=False):
    x,y = np.meshgrid(sigma_list,c_list)
    h_avg = np.zeros(np.shape(y))
    w_avg = np.zeros(np.shape(y))
    x_avg = np.zeros(np.shape(y))
    p_avg = np.zeros(np.shape(y))
    loop_area_avg = np.zeros(np.shape(y))
    n_loops_avg = np.zeros(np.shape(y))
    for sigma_i in range(len(sigma_list)):
        for c_i in range(len(c_list)):
            n_loops = 0
            num_file_successes = 0
            for n in range(max_n):
                #npzfilename = save_folder + '/c' + str(c_list[c_i]) + '_w' + str(sigma_list[sigma_i]) + '_' + str(n) +'.npz'
                #save_folder + '/c%0.2f_w%0.2f_%d.npz' % (c_list[c_i], sigma_list[sigma_i], n)
                #picklefilename = save_folder + '/pial_c' + str(c_list[c_i]) + '_w' + str(sigma_list[sigma_i]) + str(n) + '.obj'
                #save_folder + '/pial_c%0.2f_w%0.2f%d.obj' % \
                                 #(c_list[c_i],sigma_list[sigma_i],n)
                npzfilename = save_folder + '/c%0.2f_w%0.2f_%d.npz' % (c_list[c_i],sigma_list[sigma_i], n)
                picklefilename = save_folder + "/pial_c%0.2f_w%0.2f%d.obj" % \
                                   (c_list[c_i],sigma_list[sigma_i], n)
                try:
                    loaded = np.load(npzfilename)
                    h_avg[c_i, sigma_i] += np.sum(loaded['h'])
                    w_avg[c_i, sigma_i] += np.sum(loaded['w'])
                    x_avg[c_i, sigma_i] += np.sum(loaded['x'])
                    p_avg[c_i, sigma_i] += np.sum(loaded['p'])
                    loop_area_avg[c_i, sigma_i] += np.sum(np.nan_to_num(loaded['h']/loaded['x']))
                    n_loops += len(loaded['h'])
                    num_file_successes += 1
                    n_loops_avg[c_i, sigma_i] += len(loaded['h'])
                except: pass
            n_loops_avg[c_i, sigma_i] /= num_file_successes
            if n_loops != 0:
                h_avg[c_i, sigma_i] /= n_loops
                w_avg[c_i, sigma_i] /= n_loops
                x_avg[c_i, sigma_i] /= n_loops
                p_avg[c_i, sigma_i] /= n_loops
                loop_area_avg[c_i, sigma_i] /= n_loops
            '''else:
                print('no loops for s = %0.2f, c = %0.2f' % (sigma_list[sigma_i], c_list[c_i]))
                h_avg[c_i, sigma_i] = 0
                w_avg[c_i, sigma_i] = 0
                x_avg[c_i, sigma_i] = 0
                p_avg[c_i, sigma_i] = 0
                loop_area_avg[c_i, sigma_i] = 0'''
            #print(sigma_list[sigma_i], c_list[c_i], n_loops)

    sigmas = np.array(sigma_list)
    sigma_avgs = (sigmas[:-1] + sigmas[1:])/2
    sigma_bounds = [0, *sigma_avgs, sigma_list[-1]+1]

    cs = np.array(c_list)
    c_avgs = (cs[:-1] + cs[1:])/2
    c_bounds = [0, *c_avgs, c_list[-1]+1]

    xbounds, ybounds = np.meshgrid(sigma_bounds, c_bounds)

    names = ['Average number of loops in simulation', 'Average sinks per loop',
             'tree length over loop perimeter', 'PAs per area (density)',
             'tree length over loop area', 'loop area']
    datas = [n_loops_avg, h_avg, w_avg, x_avg, p_avg, loop_area_avg]
    if plot:
        for data_ind in range(len(datas)):
            meshplotter(sigma_list, c_list, np.log(datas[data_ind]),
                        title=names[data_ind], cbarlabel=names[data_ind],
                        logbool=logbool, save_folder=save_folder,
                        graphplot=True)

    return h_avg, w_avg, x_avg, p_avg, loop_area_avg, n_loops_avg

def compare_to_data(save_folder, sigma_list, c_list, max_n, expdatafiles=None,
        logbool=True, plot=True):
    metrics = ['h','w','x','p']
    labels = ['PAs per loop', 'tree length over loop perimeter',
              'PAs per area (density)', 'tree length over loop area']
    data = {}
    for i in range(len(metrics)):
        data[metrics[i]] = []
        for c_i in range(len(c_list)):
            data[metrics[i]].append([])
            for sigma_i in range(len(sigma_list)):
                data[metrics[i]][c_i].append([])

    for c_i in range(len(c_list)):
        for sigma_i in range(len(sigma_list)):
            for n in range(max_n):
                npzfilename = save_folder + '/c%0.2f_w%0.2f_%d.npz' % \
                    (c_list[c_i],sigma_list[sigma_i], n)
                picklefilename = save_folder + "/pial_c%0.2f_w%0.2f%d.obj" % \
                    (c_list[c_i],sigma_list[sigma_i], n)
                try:
                    loaded = np.load(npzfilename)
                    loadbool = True
                except: pass
                if loadbool:
                    for i in range(len(metrics)):
                        data[metrics[i]][c_i][sigma_i] = np.append(
                            data[metrics[i]][c_i][sigma_i], loaded[metrics[i]])
                    '''try:
                        with open(picklefilename, 'rb') as f:
                            pial = pickle.load(f)
                            data['numPAs'][c_i][sigma_i] = np.append(data['numPAs'][c_i][sigma_i], len(pial.penetrating_arterioles))
                            print(c_i, sigma_i)
                    except: pass'''
                loadbool = False

    ci = 7
    sigi = 13

    plt.hist(data[metrics[1]][ci][sigi], bins=16)
    plt.xlabel('tree length / loop perimeter')
    plt.ylabel('frequency')
    plt.title('c=%f, sigma=%f' % (c_list[ci], sigma_list[sigi]))
    plt.text(0.5, 7, 'N = '+str(len(data[metrics[0]][ci][sigi])))
    plt.show()

    processed = {}
    for i in range(len(metrics)):
        processed[metrics[i]] = np.zeros((len(c_list),len(sigma_list)))

    if expdatafiles == None:
        basic_entropy = True
    else:
        stats = analyze_several_data_files(expdatafiles, plot=False)

    expdensities = stats[2]
    #stats[2] = expdensities[np.logical_and(expdensities < 45, expdensities > 1e-2)]
    rho = expdensities.mean()
    lengthtomm = 1

    for i in range(len(metrics)):
        for c_i in range(len(c_list)):
            for sigma_i in range(len(sigma_list)):
                if len(data[metrics[i]][c_i][sigma_i]) > 0:
                    processed[metrics[i]][c_i, sigma_i] = ks_2samp(
                        data[metrics[i]][c_i][sigma_i], stats[i])[1]

    names = ['PAs per loop', 'tree length over loop perimeter',
             'PAs per area (density)', 'tree length over loop area']

    for data_ind in range(len(metrics)):
        meshplotter(sigma_list, c_list, 1-processed[metrics[data_ind]],
                    title='1-pval_'+names[data_ind], cbarlabel='1-pval of '+names[data_ind],
                    logbool=True, save_folder='detailed-random')

################################################################################
# New methods

def river_batch(strengths, n, savefolder):
    for _ in range(n):
        for s in strengths:
            try:
                a = make_river(s, density=65, basin_fraction=0.08,
                               shape='square', n_sinks=25, n_sources=1,
                               basins='triangle')

                nloops = a.n_cycles()
                perloop = a.count_per_loop(type='basins')
                ps = a.loop_perimeters()
                qhull_ps = a.loop_qhull_perimeters()
                areas = a.loop_areas()
                qhull_areas = a.loop_qhull_areas()
                trees = a.tree_lengths()

                x = np.array([np.zeros(nloops),
                              perloop, trees/ps,
                              ps, qhull_ps,
                              areas, qhull_areas,
                              ps/qhull_ps, areas/qhull_areas])
                x[np.isnan(x)] = 0

                try:
                    results = np.load('%s/%0.2f.npy' % (savefolder,s))
                    results = np.append(results, x.T, axis=0)
                except:
                    results = x.T

                results[0,0] += 1
                np.save('%s/%0.2f.npy' % (savefolder,s), results)
            except ZeroDivisionError:
                print('...................................Simulation failed')
                continue
            except AssertionError:
                print('...................................Loop search failed')
                continue
            except Exception as e:
                print(e)
                continue

    return results

def pial_batch(widths, strengths, n, savefolder, n_sources=10):
    try:
        with open(savefolder + '/simcounts.p','rb') as f:
            simcounts = pickle.load(f)
    except FileNotFoundError:
        simcounts = {}

    for sigma in widths:
        if sigma not in simcounts:
            simcounts[sigma] = {}
        for c in strengths:
            if c not in simcounts[sigma]:
                simcounts[sigma][c] = 0

    for _ in range(n):
        for sigma in widths:
            for c in strengths:
                try:
                    a = pi.make_pial(sigma, c, density=65, sink_fraction=0.1,
                        n_sources=n_sources)

                    perloop = a.count_per_loop()
                    ps = a.loop_perimeters()
                    qhull_ps = a.loop_qhull_perimeters()
                    areas = a.loop_areas()
                    qhull_areas = a.loop_qhull_areas()
                    trees = a.tree_lengths()

                    x = np.array([perloop, trees/ps,
                                  ps, qhull_ps,
                                  areas, qhull_areas,
                                  ps/qhull_ps, areas/qhull_areas])
                    #x[np.isnan(x)] = 0

                    try:
                        results = np.load('%s/s%0.2f_c%0.2f.npy' %
                                          (savefolder,sigma,c), allow_pickle=True)
                        results = np.append(results, x.T, axis=0)
                    except FileNotFoundError:
                        results = x.T

                    np.save('%s/s%0.2f_c%0.2f.npy' % (savefolder,sigma,c),
                            results)

                    simcounts[sigma][c] += 1

                    print(simcounts)

                    with open(savefolder+'/simcounts.p', 'wb') as f:
                        pickle.dump(simcounts, f)

                except ZeroDivisionError:
                    print('..................................Simulation failed')
                    continue
                except AssertionError:
                    print('.................................Loop search failed')
                    continue
                '''except Exception as e:
                    print(e)
                    continue'''

    return results

def pial_data_dists(files):
    i = 1
    for file in files:
        a = pi.pial_xylem(file, i)

        t = time.time()
        perloop = a.count_per_loop()
        ps = a.loop_perimeters()
        qhull_ps = a.loop_qhull_perimeters()
        areas = a.loop_areas()
        qhull_areas = a.loop_qhull_areas()
        trees = a.tree_lengths()

        x = np.array([perloop, trees/ps,
                      ps, qhull_ps,
                      areas, qhull_areas,
                      ps/qhull_ps, areas/qhull_areas])

        if i == 1:
            results = x.T
        else:
            results = np.append(results, x.T, axis=0)

        i += 1

    np.save('pial_dists.npy', results)

def pial_data_circles(file, i):
    from shapely.geometry import MultiLineString, LinearRing

    a = pi.pial_xylem(file,i)
    a.remove_trees()
    coords = [a.LEAF.Vertex[edge, :] for edge in a.LEAF.Bond]
    lines = MultiLineString(coords)

    xmax = max(a.LEAF.Vertex[:,0])
    ymax = max(a.LEAF.Vertex[:,1])

    x = np.linspace(0.2*xmax, 0.8*xmax, 10)
    y = np.linspace(0.2*ymax, 0.8*ymax, 5)
    data = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
            circ = np.zeros((len(angles),2))
            circ[:,0] = np.cos(angles) + x[i]
            circ[:,1] = np.sin(angles) + y[j]

            intersections= lines.intersection(LinearRing(circ))
            try:
                data[i,j] = len(intersections)
            except TypeError:
                data[i,j] = 1

    return data.flatten()

def pial_multidata_circles(files):
    data = np.array([])
    i = 1
    for file in files:
        a = pial_data_circles(file, i)
        data = np.append(data, a)
        i += 1

    m = np.mean(data)
    s = np.std(data)

    plt.suptitle('Network backbone points intersecting distributed circles')
    plt.title('Mean: %0.2f, St. dev.: %0.2f' % (m, s))

    plt.hist(data.flatten(), bins=np.arange(20))
    plt.show()

def pial_data_sigma(file, i):
    a = pi.pial_xylem(file, i)
    dists = a.vert_distances_sqr(verts=a.sinks)
    np.fill_diagonal(dists,100)
    return np.amin(dists, axis=0)

def pial_multidata_sigma(files):
    data = np.array([])
    i = 1
    for file in files:
        data = np.append(data, pial_data_sigma(file, i))
        i += 1
    m = np.mean(data)
    s = np.std(data)

    plt.suptitle('Minimum distances between PAs')
    plt.title('Mean: %0.4f, St. dev.: %0.4f' % (m, s))

    plt.hist(data, bins=40)
    plt.xlim([0,0.5])
    plt.show()

if __name__ == "__main__":
    #river_batch(np.linspace(0,3,31), 100, 'riverdata4.npy')

    pial_files = ['MDA101L_20170520_144817.mat',
                    'MDA105L_20170522_105334.mat',
                    'MDA106L_20170522_110804.mat',
                    'MDA302L_20170522_134110.mat',
                    'MDA303L_20170522_135724.mat',
                    'MDA304L_20170522_140157.mat',
                    'MDA305L_20170522_141712.mat',
                    'MDA401L_20170522_112005.mat',
                    'MDA402L_20170522_113536.mat',
                    'MDA403L_20170522_114900.mat',
                    'MDA404L_20170522_142801.mat',
                    'MDA405L_20170522_143707.mat',
                    'MDA407L_20170522_120231.mat',
                    'MDA503L_20170522_145147.mat',
                    'MDA601L_20170522_150013.mat',
                    'MDA702L_20170522_121314.mat',
                    'MDA704L_20170522_151851.mat',]

    pial_data_dists(pial_files)

    '''sigma_list = [0.1 ,0.2, 0.5, 1, 2, 5, 10]
    c_list = [0.1 ,0.2, 0.5, 1, 2, 5, 10, 20, 50]
    #study('3source', sigma_list, c_list, 10, 15)
    #batch_analysis('3source', sigma_list, c_list, 10)'''

    '''sigma_list = np.linspace(0.1, 5, 10)
    c_list = np.logspace(-1.5, 1.5, 8)
    #study('loglin3source', sigma_list, c_list, 1, 5)
    batch_analysis('loglin3source', sigma_list, c_list, 5, logbool=True)'''

    '''sigma_list = np.linspace(0.01, 5, 15)
    c_list = np.logspace(-1.5, 2, 15)
    study('detailed', sigma_list, c_list, 5, 10)
    #batch_analysis('detailed', sigma_list, c_list, 5, logbool=False)'''

    sigma_list = np.linspace(0.01, 5, 15)
    c_list = np.logspace(-1.5, 2, 15)
    #study('detailed-random', sigma_list, c_list, 14, 25, min_sigma=2,c_bounds=[0.5,9])
    #batch_analysis('detailed-random', sigma_list, c_list, 25, logbool=False, plot=True,)
    #batch_analysis('detailed-random', sigma_list, c_list, 22, plot=True,)

    #analyze_several_data_files(all_data_files, plot=False)

    #compare_to_data('detailed-random', sigma_list, c_list, 25, expdatafiles=all_data_files)
