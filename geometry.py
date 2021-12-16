import math

from numpy import *

import LEAFclass as LFCLSS

"""
    geometry.py

    Contains some functions related to the geometry of
    the system, for example lattice bounding polygons etc...
"""

def xscale(LEAF, f):
    """
        Scales the vertices of LEAF by the factor f in 
        x direction.
    """
    LEAF.Vertex[:,0] *= f

def leafshape(angle):
    """
        Generates a LeafShape with the given angle
        such that aheight = awidth*sin(angle)
    """
    theta = arange(0.,7.) * pi/3. + pi
    LSHP = LFCLSS.LeafShape('Leaf Shape', angle)
    LSHP.polyedge = array([cos(theta)*1.21+1.2, sin(theta)*1.21])
    beta = math.atan(2*sin(angle))

    c = sqrt((LSHP.polyedge[0,0] - LSHP.polyedge[0,1])**2 + \
        (LSHP.polyedge[1,0] - LSHP.polyedge[1,1])**2)
    h = LSHP.polyedge[1,1] - LSHP.polyedge[1,0]

    dx = LSHP.polyedge[0,1] - (LSHP.polyedge[0,0] + sqrt((h/sin(beta))**2 - \
        (LSHP.polyedge[1,0] - LSHP.polyedge[1,1])**2))
    
    LSHP.polyedge[0, 1] -= dx
    LSHP.polyedge[0, 2] += dx
    LSHP.polyedge[0, 4] += dx
    LSHP.polyedge[0, 5] -= dx
    
    return LSHP

def poly_bounds(polyedge):
    """
        returns width, height of a bounding rectangle around
        the polygon defined by polyedge
    """
    
    width = max(polyedge[0]) - min(polyedge[0]) + 0.1
    height = max(polyedge[1]) - min(polyedge[1]) + 0.1

    return width, height

def bounding_box(polyedge):
    """
        Calculates and returns the minimum bounding
        rectangle of the given polygon.
        The bounding box is oriented along the coordinate axes.
    """

    top_left = array([min(polyedge[0]), max(polyedge[1])])
    top_right = array([max(polyedge[0]), max(polyedge[1])])
    bot_left = array([min(polyedge[0]), min(polyedge[1])])
    bot_right = array([max(polyedge[0]), min(polyedge[1])])

    width = linalg.norm(top_left - top_right) + 0.1
    height = linalg.norm(bot_left - top_left) + 0.1

    return bot_left, top_right, width, height

def subdivide_bonds(LEAF):
    """
        Subdivides each bond in the leaf described by LEAF
        into 2 bonds.

        returns a list of the new vertices at the centers
        of old bonds,
        a list of bond pairs making up the old bonds,
        and a list of bond indices pointing to the old bond
        for each new vertex
    """
    vnum = len(LEAF.Vertex)
    bnum = len(LEAF.Bond)
    new_bonds = list(LEAF.Bond.copy())
    new_verts = []
    pairs = []
    old_bonds = []
    new_vert_indices = []

    for i in xrange(len(LEAF.Bond)):
        # Find vertex in the middle
        vnew = 0.5*(LEAF.Vertex[LEAF.Bond[i,0]] + LEAF.Vertex[LEAF.Bond[i,1]])
        
        # New connectivity
        new_verts.append(vnew)

        old_target = LEAF.Bond[i,1]
        new_bonds[i][1] = vnum
        new_bonds.append(array([vnum, old_target]))
        pairs.append([i, bnum])
        old_bonds.append(i)
        new_vert_indices.append(vnum)

        vnum += 1
        bnum += 1

    LEAF.Bond = array(new_bonds)
    LEAF.Vertex = vstack((LEAF.Vertex, array(new_verts)))

    return array(new_vert_indices), array(pairs), array(old_bonds)
