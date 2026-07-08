import pyscf
import numpy as np
import pdb
from gpu4pyscf.dft import rks
def check_orbital_orders(atoms, pos, Ham, Ham_init):
    single_mol = pyscf.gto.Mole()
    single_mol.atom = [[atoms[i], pos[i]] for i in range(len(atoms))]
    single_mol.basis = 'def2-svp'
    single_mol.build()
    mf = pyscf.scf.RKS(single_mol).set(xc='pbe')
    dm_minao = mf.init_guess_by_minao()
    Ham_minao = mf.get_fock(dm=dm_minao)
    mf.kernel()
    Ham_pyscf = mf.get_fock()
    
    
    print("=" * 50, flush=True)
    print(Ham, flush=True)
    print(Ham_pyscf, flush=True)
    print(np.allclose(Ham, Ham_pyscf, atol=1e-3), flush=True) # check if the order of orbital is correct
    # print(np.allclose(Ham_init, Ham_minao, atol=1e-3), flush=True) # check if the order of orbital is correct
    print("=" * 50)

    return