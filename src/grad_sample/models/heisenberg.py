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

class J1J2:
    def __init__(self, L=3, J=[1.0,0.5], sign_rule=[False,False], acting_on_subspace=0):
        self.name = "J1J2"
        self.Ns = L*L
        self.L = L
        self.h = J[1]
        self.sign_rule = sign_rule[0]
        self.acting_on_subspace = acting_on_subspace

        self.lattice = nk.graph.Square(L, max_neighbor_order=len(J), pbc=True)
        
        self.hi = nk.hilbert.Spin(s=1/2, N=self.lattice.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=J, sign_rule=sign_rule, acting_on_subspace=self.acting_on_subspace)
        
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

class XXZ:
    def __init__(self, L=10, h=1.5):
        self.name = "xxz"
        self.Ns = L
        self.L = L
        self.h = h
        
        self.lattice = nk.graph.Chain(L, pbc=True)
        
        self.hi = nk.hilbert.Spin(s=1/2, N=self.lattice.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.H = 0
        for i in range(L):
            self.H += nk.operator.spin.sigmax(self.hi,i)*nk.operator.spin.sigmax(self.hi,(i+1)%self.L) + nk.operator.spin.sigmay(self.hi,i)*nk.operator.spin.sigmay(self.hi,(i+1)%self.L) + self.h*(nk.operator.spin.sigmaz(self.hi,i)*nk.operator.spin.sigmaz(self.hi,(i+1)%self.L))
        # op = nk.operator.GraphOperator(self.hi, graph=self.lattice, bond_ops=bond_operator)
        # self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)
        
        self.H_jax = self.H.to_jax_operator()

        self.H_sp = self.H.to_sparse()
        