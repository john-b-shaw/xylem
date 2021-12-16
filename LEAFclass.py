class LeafShape:
    def __init__(self, comment, angle):
        self.comment = comment
        self.polyedge = []
        self.width = []
        self.nsharp = []
        self.basewidth = []
        self.baselength = []
        self.aspectr = []
        self.q = []
        self.angle = angle

class Topology:
    def __init__(self, comment, lattice):
        self.comment = comment
        self.lattice = lattice
        self.height = []
        self.Vertex = []
        self.Neigh = []
        self.Bond = []

class LeafEnergy:
    def __init__(self,comment):
        self.comment = comment
        self.vectorConductivity = []
        self.Concentrations = []
        self.Currents = []
        self.CurrentMatrix = []
        self.alpha = []
        self.Energy = []
        self.gamma = []
        self.Pressures = []
