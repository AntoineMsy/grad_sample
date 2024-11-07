import netket as nk

class TFI:
    def __init__(self, L=3, h=1.0):
        self.Ns = L * L
        self.h = h
        self.L = L
        self.name = "ising"
        # lattice = nk.graph.Square(L, max_neighbor_order=2)

        self.lattice = nk.graph.Square(L, pbc=True)
        
        self.hi = nk.hilbert.Spin(s=1 / 2, N=self.lattice.n_nodes, inverted_ordering=False)
        
        self.H = nk.operator.Ising(hilbert=self.hi, graph=self.lattice, h=self.h)
        
        self.H_jax = self.H.to_jax_operator()

        self.H_sp = self.H.to_sparse()