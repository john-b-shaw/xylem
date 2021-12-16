# xylem
Investigate complex channel network morphodynamics under hydrodynamic fluctuations

Here is a readme prepared by Adam Konkol on November 8, 2021

analyze.py: primarily for analysing pial vascular simulations
geometry.py: legacy code, I never used it
InitFunctions.py: I think the only function I used from this is Create_networkFO, which was used to define the lattice that the network is on. This became obsolete because I moved the code to static method make_LEAF in class NetworkSuite (see xylem.py)
LEAFclass.py: defines classes Topology (LEAF) and LeafShape (LSHP), which are the backbone of the higher order classes in xylem.py
network.py: functions at the top of the file are legacy code/never used. The higher order classes in xylem.py are built on VascularNetwork class, which provides some of the basic graph/network methods. The methods for conductivity, laplacian, adjacency matrices and current vector are useful. I also changed the plot_conductivities_raw function for use in the classes of xylem.py
plotting.py: various plotting functions for networks and statistics after they were simulated
simulate.py: stores the functions called to run batch simulations (so the details of the simulations that were used in the paper are in these functions) and some functions to analyse the simulated networks once they were done
stats.py: one function to provide a common interface for getting statistics out of a given network. used in the other files
storage.py: never used, the save function in the xylem.py classes should be used instead
xylem.py: workhorse of the code. The NetworkSuite class defines the common methods and attributes for defining a network and simulating its evolution, trimming edges for analysis, studying the topology, and plotting. The file also contains the PialNetwork class for the brain vascular simulations, DeltaNetwork class for this project, and SeedNetwork class for something you asked for one time. There are some functions at the bottom of the file useful for loading Jon's data into the DeltaNetwork class so it can be used for analysis.
