import numpy as np
import warnings

def stat(a, file, thr=1e-6):
    if a in ['Ras Isa', 'Sarawak']:
        return 0
    if file == 'wbridges':
        """ Returns fraction of weighted channel length contained in loops.
            Channels are weighted by their width, so the statistic is
            the channel surface area of all deltaic channels that comprises
            loops
        """
        with warnings.catch_warnings(record=True) as w:
            x = 1-a.bridges(weight=True, smooth=False, thr=thr)
            if len(w) > 0:
                x = 0
        delattr(a, 'G')
    elif file == 'bridges':
        """ Returns fraction of channel length (unweighted) contained in loops.
        """
        x = 1-a.bridges(weight=False, smooth=True, thr=thr)
        delattr(a, 'G')
    elif file == 'nloops':
        """ returns the number of loops in the network
        """
        x = a.n_cycles(thr=thr)#/a.area
        delattr(a, 'G')
    elif file == 'mstdiff':
        """ Returns the fraction of channel area (as weighted in wbridges)
            that has to be removed to make the minimum spanning tree.
            mstdiff = Minimum Spanning Tree DIFFerence
        """
        x = a.mstdiff(thr=thr)
        delattr(a, 'G')
    elif file == 'mstdiffl':
        """ returns fraction of channel length removed to make minimum spanning
            tree. To mstdiff as bridges is to wbridges
        """
        x = a.mstdiff(thr=thr, weight=False)
        delattr(a, 'G')
    elif file == 'resdist':
        """ Returns two lists. The first one of resistance distances from
            river apex (there can only be one for this method) to the sinks,
            the second from all of the tidal nodes to the sinks
        """
        x, y = a.resistance_distances()
    elif file == 'loopareas':
        """ Delta plane area encompassed by loops divided by delta convex hull
            area
        """
        x = np.sum(a.loop_areas(thr=thr)) / a.area
        delattr(a, 'G')
        delattr(a, 'cycles')
    elif file == 'pathnodes':
        """ returns a list of the number of paths from the source node to each
            of the sink nodes. slow
        """
        x = a.path_nodes()
    elif file == 'flowchange':
        """ fraction of channels that change flow direction over tidal ensemble
        """
        x = a.flow_change()
    return x
