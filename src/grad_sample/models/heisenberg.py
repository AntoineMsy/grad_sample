import netket as nk

class Heisenberg2d:
    def __init__(self, L=3, J=1.0, sign_rule=False, acting_on_subspace=0):
        self.name = "heisenberg2d"
        self.Ns = L*L
        self.L = L
        self.h = J
        self.sign_rule = sign_rule
        self.acting_on_subspace = acting_on_subspace

        self.lattice = nk.graph.Square(L, pbc=True)
        
        self.hi = nk.hilbert.Spin(s=1/2, N=self.lattice.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)
        
        self.H_jax = self.H.to_jax_operator()

        self.H_sp = self.H.to_sparse()


class Heisenberg1d:
    def __init__(self, L=3, J=1.0, sign_rule=False, acting_on_subspace=0):
        self.name = "heisenberg1d"
        self.Ns = L
        self.L = L
        self.h = J
        self.sign_rule = sign_rule
        self.acting_on_subspace = acting_on_subspace

        self.lattice = nk.graph.Chain(L, pbc=True)
        
        self.hi = nk.hilbert.Spin(s=1/2, N=self.lattice.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)
        
        self.H_jax = self.H.to_jax_operator()

        self.H_sp = self.H.to_sparse()