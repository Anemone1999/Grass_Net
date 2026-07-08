from argparse import Namespace
import numpy as np
CONVENTION_DICT = {
        'def2-qzvp': Namespace(
            atom_to_orbitals_map={1: 'sssspppddf', 6: 'sssssssppppdddffg', 7: 'sssssssppppdddffg', 8: 'sssssssppppdddffg',
                                9: 'sssssssppppdddffg'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+7*1,"d":0+7*1+4*3,"f":0+7*1+4*3+3*5,"g":0+7*1+4*3+3*5+2*7},
            max_block_size= 57,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array([0, 1, 2, 3, 4]), 'f': np.array([0, 1, 2, 3, 4, 5, 6]),
                             'g':np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])},
        ),
        'def2-tzvp': Namespace(
            atom_to_orbitals_map={1: 'sssp', 6: 'ssssspppddf', 7: 'ssssspppddf', 8: 'ssssspppddf',
                                9: 'ssssspppddf'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+5*1,"d":0+5*1+3*3,"f":0+5*1+3*3+2*5},
            max_block_size= 31,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array( [0, 1, 2, 3, 4]), 'f': np.array([0, 1, 2, 3, 4, 5, 6])},
        ),
        'def2-svp': Namespace(
            atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd',
                                9: 'sssppd'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+3*1,"d":0+3*1+2*3},
            max_block_size= 14,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array( [0, 1, 2, 3, 4])},
        ),
        'def2-svp-nabladft': Namespace(
            atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd', 
                              16: 'sssspppd', 17: 'sssspppd', 35: 'sssssppppddd'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+3*1,"d":0+3*1+2*3},
            max_block_size= 32,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array( [0, 1, 2, 3, 4])},
        ),
    }