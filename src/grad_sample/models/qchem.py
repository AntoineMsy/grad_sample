import netket as nk
import pubchempy as pcp
from pyscf import gto, scf, fci, cc, mcscf
import netket.experimental as nkx
from netket_pro.operator import ParticleNumberConservingFermioperator2ndSpinJax

class PCMolecule:
    """
    Initialize a model from a PubChem CID
    Defaults
    """
    def __init__(self, cid=947):
        # 1. Load the N2 molecule from PubChem (3D structure)
        self.cid = cid  # N2 molecule
        compound = pcp.get_compounds(cid, "cid", record_type="3d")[0]
        self.name = cid
        # 2. Extract atomic coordinates
        geometry = []
        for atom in compound.atoms:
            symbol = atom.element
            x, y, z = atom.x, atom.y, atom.z
            geometry.append(f"{symbol} {x} {y} {z}")

        # Convert to PySCF format
        mol_geometry = "\n".join(geometry)

        # 3. Define the molecule in PySCF
        mol = gto.Mole()
        mol.atom = mol_geometry
        mol.basis = "STO-3G"  # Choose a reasonable basis set
        mol.unit = "angstrom"  # Coordinates are in Ångströms
        mol.spin = 0  # N2 is a singlet
        mol.charge = 0
        mol.build()

        # 4. Run Hartree-Fock calculation
        mf = scf.RHF(mol)
        mf.kernel()

        # 5. Compute Full Configuration Interaction (FCI) energy 
        ccsd = cc.ccsd.CCSD(mf).run()
        nat_orbs = mcscf.addons.make_natural_orbitals(ccsd)

        # natorbital hamiltonian
        ha_pyscf = nkx.operator.from_pyscf_molecule(mol, mo_coeff=nat_orbs[1]).to_jax_operator()
        self.hamiltonian = ha_pyscf
        # self.hamiltonian = ParticleNumberConservingFermioperator2ndSpinJax.from_fermiop(ha_pyscf)
        self.hilbert_space = self.hamiltonian.hilbert
        
        g = nk.graph.Chain(self.hilbert_space.n_orbitals, pbc=False)
        self.graph = nk.graph.disjoint_union(g, g) #only relevant for the fermihop sampler
        
        # define dummy vars for automatic naming
        self.Ns = self.hilbert_space.size
        self.h = 0
        self.L = self.Ns

        # try:
        # 5. Compute Full Configuration Interaction (FCI) energy
        cisolver = fci.FCI(mol, mf.mo_coeff)
        self.E_fci = cisolver.kernel()[0]
        # except:
        #     self.E_fci = None
