import netket as nk
from netket.utils.types import Array
from deepnets.system.base import Spin_Half
import jax.numpy as jnp
class Heisenberg2d:
    def __init__(self, L=3, J=1.0, sign_rule=False, acting_on_subspace=0):
        self.name = "heisenberg2d"
        self.Ns = L*L
        self.L = L
        self.h = J
        self.sign_rule = sign_rule
        self.acting_on_subspace = acting_on_subspace

        self.graph = nk.graph.Square(L, pbc=True)
        
        self.hilbert_space = nk.hilbert.Spin(s=1/2, N=self.graph.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.hamiltonian = nk.operator.Heisenberg(hilbert=self.hilbert_space, graph=self.graph, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)
        
        # self.H_jax = self.H.to_jax_operator()

class J1J2:
    def __init__(self, L=3, J=[1.0,0.5], sign_rule=[False,False], acting_on_subspace=0):
        self.name = "J1J2"
        self.Ns = L*L
        self.L = L
        self.J = J
        self.h = J[1]
        self.sign_rule = sign_rule[0]
        self.acting_on_subspace = acting_on_subspace

        self.graph = nk.graph.Square(L, max_neighbor_order=len(J), pbc=True)
        
        self.hilbert_space = nk.hilbert.Spin(s=1/2, N=self.graph.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.hamiltonian = nk.operator.Heisenberg(hilbert=self.hilbert_space, graph=self.graph, J=J, sign_rule=sign_rule, acting_on_subspace=self.acting_on_subspace)

class Heisenberg1d:
    def __init__(self, L=3, J=1.0, sign_rule=False, acting_on_subspace=0):
        self.name = "heisenberg1d"
        self.Ns = L
        self.L = L
        self.h = J
        self.sign_rule = sign_rule
        self.acting_on_subspace = acting_on_subspace

        self.graph = nk.graph.Chain(L, pbc=True)
        
        self.hilbert_space = nk.hilbert.Spin(s=1/2, N=self.lattice.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.hamiltonian = nk.operator.Heisenberg(hilbert=self.hi, graph=self.graph, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)

class XXZ:
    def __init__(self, L=10, h=1.5):
        
        self.name = "xxz"
        self.Ns = L
        self.L = L
        self.h = h
        
        self.graph = nk.graph.Chain(L, pbc=True)
        
        self.hilbert_space = nk.hilbert.Spin(s=1/2, N=self.graph.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.hamiltonian = 0
        for i in range(L):
            self.hamiltonian += nk.operator.spin.sigmax(self.hilbert_space,i)*nk.operator.spin.sigmax(self.hilbert_space,(i+1)%self.L) + nk.operator.spin.sigmay(self.hilbert_space,i)*nk.operator.spin.sigmay(self.hilbert_space,(i+1)%self.L) + self.h*(nk.operator.spin.sigmaz(self.hilbert_space,i)*nk.operator.spin.sigmaz(self.hilbert_space,(i+1)%self.L))
        # op = nk.operator.GraphOperator(self.hi, graph=self.lattice, bond_ops=bond_operator)
        # self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)

class XXZ2d(Spin_Half):

    def __init__(self, L=4, h=1.5):
        super().__init__(N=int(L**2), L=L, J=jnp.array([h]), sz_sector=0.0)
        self.name = "xxz"
        self.Ns = L
        self.L = L
        self.h = h
        
        self.graph = nk.graph.Square(L, pbc=True)
        
        # self.hilbert_space = nk.hilbert.Spin(s=1/2, N=self.graph.n_nodes, total_sz=0, inverted_ordering=False)
        
        self.hamiltonian = 0
        for (i,j) in self.graph.edges:
            self.hamiltonian += nk.operator.spin.sigmax(self.hilbert_space,i)*nk.operator.spin.sigmax(self.hilbert_space,j) + nk.operator.spin.sigmay(self.hilbert_space,i)*nk.operator.spin.sigmay(self.hilbert_space,j) + self.h*(nk.operator.spin.sigmaz(self.hilbert_space,i)*nk.operator.spin.sigmaz(self.hilbert_space,j))
        # op = nk.operator.GraphOperator(self.hi, graph=self.lattice, bond_ops=bond_operator)
        # self.H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.lattice, J=self.h, sign_rule=self.sign_rule, acting_on_subspace=self.acting_on_subspace)
        
    @staticmethod
    def extract_patches_as1d(x, b):
        """
        Extract bxb patches from the (nbatches, nsites) input x.
        Returns x reshaped to (nbatches,npatches,patch_size), where patch_size = b**2
        """
        batch = x.shape[0]
        L_eff = int((x.shape[1] // b**2) ** 0.5)
        x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
        x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
        # flatten the patches
        x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
        x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
        return x

    @staticmethod
    def extract_patches_as2d(x: Array, b: int, lattice_shape=None) -> Array:
        """
        Extract bxb patches from the (nbatches, nsites) input x, lattice_shape is a dummy argument so the call has a consistent pattern
        across systems.
        Returns x reshaped to (nbatches,x,y,patch_size) where x,y are x,y coordinates of the patch_size = b**2 patches
        For the square lattice with site indexing I = j + i*Ly.
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        batch = x.shape[0]
        L_eff = int((x.shape[-1] // b**2) ** 0.5)
        x = x.reshape(batch, L_eff, b, L_eff, b)
        x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
        x = x.reshape(batch, L_eff, L_eff, b**2)  # [L_eff, L_eff, b*b]
        return x

    @staticmethod
    def reshape_xy(x: Array, lattice_shape: tuple) -> Array:
        """
        Reshape a (nbatch, nsites) array into an (nbatch,x,y,1) array, which can then have a 2d convolutional layer applied, where (x,y) label the real space coordinates of the point
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        x = x.reshape(
            (x.shape[0], lattice_shape[0], lattice_shape[1])
        )  # (nbatch, x, y)
        x = x.reshape(x.shape + (1,))  # shape (nbatch, x, y, 1)
        return x
    