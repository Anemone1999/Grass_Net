from argparse import Namespace
import numpy as np
import gc
import copy

from src.utility.constant import CONVENTION_DICT

def get_conv_variable(basis = "def2-tzvp"):
    # str2order = {"s":0,"p":1,"d":2,"f":3}
    chemical_symbols = ["n", "H", "He" ,"Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
            "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", 
            "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", 
            "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
            "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", 
            "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

    #orbital reference (for def2tzvp basis)
    orbitals_ref = {}
    mask = {}
    if basis == "def2-tzvp":
        orbitals_ref[1]  = np.array([0,0,0,    1])                   #H: 2s 1p
        orbitals_ref[6]  = np.array([0,0,0,0,0,1,1,1,    2,2,3])     #C: 3s 2p 1d
        orbitals_ref[7]  = np.array([0,0,0,0,0,1,1,1,    2,2,3])     #N: 3s 2p 1d
        orbitals_ref[8]  = np.array([0,0,0,0,0,1,1,1,    2,2,3])     #O: 3s 2p 1d
        orbitals_ref[9]  = np.array([0,0,0,0,0,1,1,1,    2,2,3])     #F: 3s 2p 1d
        orbitals_ref[15] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,3]) #P
        orbitals_ref[16] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,3]) #S
        orbitals_ref[17] = np.array([0,0,0,0,0,1,1,1,1,1,2,2,3]) #Cl
                                    #0,1,2,3,4,(5,6,7)(8,9,10)(11,12,13)(14,15,16)(17,18,19)(20,21,22,23,24)(25,26,27,28,29)(30,31,32,33,34,35,36)
        mask[1] = np.array([0,1,2,5,6,7])
        mask[6] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[7] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[8] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[9] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[15] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[16] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        mask[17] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])

    return CONVENTION_DICT[basis], orbitals_ref, mask,chemical_symbols
def matrix_transform(hamiltonian, atoms, conv):

    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new

def split2blocks(H, C, Z, orbitals_ref, mask, block_size):
    local_orbitals = []
    local_orbitals_number = 0
    Z = Z.reshape(-1)
    for z in Z:
        local_orbitals.append(
            tuple((int(z), int(l)) for l in orbitals_ref[z.item()])
        )
        local_orbitals_number += sum(2 * l + 1 for _, l in local_orbitals[-1])

    orbitals = local_orbitals

    # atom2orbitals = dict()
    Norb = 0
    begin_index_dict = []
    end_index_dict = []
    for i in range(len(orbitals)):
        begin_index_dict.append(Norb)
        for z, l in orbitals[i]:
            Norb += 2 * l + 1
        end_index_dict.append(Norb)
        # if z not in atom2orbitals:
        #     atom2orbitals[z] = orbitals[i]

    # max_len = max(len(orbit) for orbit in atom2orbitals.values())  
    # atom_orb = np.zeros((max(atom2orbitals.keys())+1, max_len), dtype=np.int64)
    # for key, value in atom2orbitals.items():  
    #     atom_orb[key, :len(value)] = np.array([it[1] for it in value], dtype=np.int64)

    # block_size = 37#np.int32((atom_orb*2+1).sum(axis=1).max())
    matrix_diag = np.zeros(
        (Z.shape[0], block_size, block_size), dtype=np.float32
    )
    matrix_non_diag = np.zeros(
        (Z.shape[0]*(Z.shape[0]-1), block_size, block_size), dtype=np.float32
    )
    matrix_diag_init = np.zeros(
        (Z.shape[0], block_size, block_size), dtype=np.float32
    )
    matrix_non_diag_init = np.zeros(
        (Z.shape[0]*(Z.shape[0]-1), block_size, block_size), dtype=np.float32
    )
    mask_diag = np.zeros(
        (Z.shape[0], block_size, block_size), dtype=np.float32
    )
    mask_non_diag = np.zeros(
        (Z.shape[0]*(Z.shape[0]-1), block_size, block_size), dtype=np.float32
    )
    non_diag_index = 0
    for i in range(len(orbitals)):  # loop over rows
        for j in range(len(orbitals)):  # loop over columns
            z1 = orbitals[i][0][0]
            z2 = orbitals[j][0][0]
            mask1 = mask[z1]
            mask2 = mask[z2]
            if i==j:
                subblock_H = H[
                        begin_index_dict[i] : end_index_dict[i],
                        begin_index_dict[j] : end_index_dict[j],
                    ]
                subblock_C = C[
                        begin_index_dict[i] : end_index_dict[i],
                        begin_index_dict[j] : end_index_dict[j],
                    ]
                matrix_diag[i][np.ix_(mask1,mask2)] = subblock_H
                matrix_diag_init[i][np.ix_(mask1,mask2)] = subblock_C
                mask_diag[i] = matrix_diag[i] != 0
            
            else:
                subblock_H = H[
                        begin_index_dict[i] : end_index_dict[i],
                        begin_index_dict[j] : end_index_dict[j],
                    ]
                subblock_C = C[
                        begin_index_dict[i] : end_index_dict[i],
                        begin_index_dict[j] : end_index_dict[j],
                    ]
                matrix_non_diag[non_diag_index][np.ix_(mask1,mask2)] = subblock_H
                matrix_non_diag_init[non_diag_index][np.ix_(mask1,mask2)] = subblock_C
                mask_non_diag[non_diag_index] = matrix_non_diag[non_diag_index] != 0
                non_diag_index +=1
    return matrix_diag, matrix_non_diag, matrix_diag_init, matrix_non_diag_init, mask_diag, mask_non_diag
    

from argparse import Namespace
#pyscf px py pz
#tp: py pz px
STR2ORDER = {"s":0,"p":1,"d":2,"f":3}

CHEMICAL_SYMBOLS = ["n", "H", "He" ,"Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
            "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", 
            "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", 
            "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
            "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", 
            "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
 
def get_conv_variable_lin(basis = "def2-tzvp"):
    # str2order = {"s":0,"p":1,"d":2,"f":3}
    conv = CONVENTION_DICT[basis]
    mask = {}
    for atom in conv.atom_to_orbitals_map:
        mask[atom] = []
        orb_id = 0
        visited_orbital = set()
        for s in conv.atom_to_orbitals_map[atom]:
            if s not in visited_orbital:
                visited_orbital.add(s)
                orb_id = conv.str2idx[s]

            mask[atom].extend(conv.orbital_idx_map[s]+orb_id)
            orb_id += len(conv.orbital_idx_map[s])
    for key in mask:
        mask[key] = np.array(mask[key])
    return conv, None, mask,None

def matrixtoblock_lin(H,Z,mask_lin,max_block_size,sym = False):
    """_summary_

    Args:
        H (_type_): _description_
        Z (_type_): _description_
        mask_lin (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_atom = len(Z)
    Z = Z.reshape(-1)
    new_H = np.zeros((n_atom*max_block_size,n_atom*max_block_size),dtype = np.float32)
    new_mask = np.zeros((n_atom*max_block_size,n_atom*max_block_size),dtype = np.float32)
    atom_orbitals = []
    for i in range(n_atom):
        atom_orbitals.append(i*max_block_size+mask_lin[Z[i]])
    atom_orbitals = np.concatenate(atom_orbitals,axis= 0)
    new_H_tmp = np.zeros((n_atom*max_block_size,len(atom_orbitals)))
    new_mask_tmp = np.zeros((n_atom*max_block_size,len(atom_orbitals)))

    new_mask_tmp[atom_orbitals] = 1
    new_mask[:,atom_orbitals] = new_mask_tmp
    new_mask = new_mask.reshape(n_atom,max_block_size,n_atom,max_block_size)
    new_mask = new_mask.transpose(0,2,1,3)

    # new_H[atom_orbitals][:,atom_orbitals] = H
    
    new_H_tmp[atom_orbitals] = H
    new_H[:,atom_orbitals] = new_H_tmp
    ## the part above is to transform H to the orbital order given in mask_lin, which is obtained from CONVENTION_DICT[basis]
    ## the part below is to cut H to blocks
    new_H = new_H.reshape(n_atom,max_block_size,n_atom,max_block_size)
    new_H = new_H.transpose(0,2,1,3)
    if sym:
        unit_matrix = np.ones((n_atom,n_atom))
        # if up and down remove eye
        upper_triangular_matrix = unit_matrix - np.triu(unit_matrix)
        diag = new_H[np.eye(n_atom)==1]
        non_diag = new_H[upper_triangular_matrix==1]
        
        diag_mask = new_mask[np.eye(n_atom)==1]
        non_diag_mask = new_mask[upper_triangular_matrix==1]
        del new_H,new_mask,new_H_tmp,new_mask_tmp

        return diag,non_diag,diag_mask,non_diag_mask
    else:
        unit_matrix = np.ones((n_atom,n_atom))
        # if up and down remove eye
        upper_triangular_matrix = unit_matrix - np.eye(len(Z))
        # # if up remove eye
        # upper_triangular_matrix = np.triu(unit_matrix) - np.eye(len(Z))
        diag = new_H[np.eye(n_atom)==1]
        non_diag = new_H[upper_triangular_matrix==1]
        
        diag_mask = new_mask[np.eye(n_atom)==1]
        non_diag_mask = new_mask[upper_triangular_matrix==1]
        del new_H,new_mask,new_H_tmp,new_mask_tmp

        return diag,non_diag,diag_mask,non_diag_mask


import torch

# def block2matrix(Z,diag,non_diag,mask_lin,max_block_size,sym = False):
#     if isinstance(Z,torch.Tensor):
#         if not isinstance(mask_lin[1],torch.Tensor):
#             for key in mask_lin:
#                 mask_lin[key] = torch.from_numpy(mask_lin[key])
#         Z = Z.reshape(-1)
#         n_atom = len(Z)
#         atom_orbitals = []
#         for i in range(n_atom):
#             atom_orbitals.append(i*max_block_size+mask_lin[Z[i].item()])
#         atom_orbitals = torch.cat(atom_orbitals,dim= 0)

#         rebuild_fock = torch.zeros((n_atom,n_atom,max_block_size,max_block_size)).to(Z.device)
        
        
#         if sym:
#             ## down
#             rebuild_fock[torch.eye(n_atom)==1] = diag
#             unit_matrix = torch.ones((n_atom,n_atom))
#             down_triangular_matrix = unit_matrix- torch.triu(unit_matrix)
#             # rebuild_fock[down_triangular_matrix==1] = 2*non_diag
#             index = 0 
#             for j in range(n_atom):  # 列  
#                 for i in range(j + 1, n_atom):  # 行，不包括对角线 
#                     if index < non_diag.shape[0]:  # 确保索引不超出源张量的范围  
#                         rebuild_fock[i, j] = non_diag[index]*2  
#                         index += 1  
#             rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
#         else:
#             # no sym
#             rebuild_fock[torch.eye(n_atom)==1] = diag
#             unit_matrix = torch.ones((n_atom,n_atom))
#             matrix_noeye = unit_matrix - torch.eye(len(Z))
#             # rebuild_fock[matrix_noeye==1] = non_diag
#             index = 0 
#             for j in range(n_atom):  # 列  
#                 for i in range(n_atom):  # 行，不包括对角线 
#                     if i == j:
#                         continue
#                     if index < non_diag.shape[0]:  # 确保索引不超出源张量的范围  
#                         rebuild_fock[i, j] = non_diag[index]
#                         index += 1  
#             rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
        
#         rebuild_fock = torch.permute(rebuild_fock,(0,2,1,3))
#         rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
#         rebuild_fock = rebuild_fock#[atom_orbitals][:,atom_orbitals]
#         return rebuild_fock
        
#     else:
#         Z = Z.reshape(-1)
#         n_atom = len(Z)
#         atom_orbitals = []
#         for i in range(n_atom):
#             atom_orbitals.append(i*max_block_size+mask_lin[Z[i]])
#         atom_orbitals = np.concatenate(atom_orbitals,axis= 0)
#         rebuild_fock = np.zeros((n_atom,n_atom,max_block_size,max_block_size))
        
        
#         if sym:
#             ## down
#             rebuild_fock[np.eye(n_atom)==1] = diag
#             unit_matrix = np.ones((n_atom,n_atom))
#             down_triangular_matrix = unit_matrix- np.triu(unit_matrix)
#             rebuild_fock[down_triangular_matrix==1] = 2*non_diag
#             rebuild_fock = (rebuild_fock + rebuild_fock.transpose(1,0,3,2))/2
#         else:
#             # no sym
#             rebuild_fock[np.eye(n_atom)==1] = diag
#             unit_matrix = np.ones((n_atom,n_atom))
#             matrix_noeye = unit_matrix - np.eye(len(Z))
#             rebuild_fock[matrix_noeye==1] = non_diag
        
#         rebuild_fock = rebuild_fock.transpose(0,2,1,3)
#         rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
#         rebuild_fock = rebuild_fock[atom_orbitals][:,atom_orbitals]
#         return rebuild_fock

def block2matrix(Z,diag,non_diag,mask_lin,max_block_size,sym = False):
    if isinstance(Z,torch.Tensor):
        if not isinstance(mask_lin[1],torch.Tensor):
            for key in mask_lin:
                mask_lin[key] = torch.from_numpy(mask_lin[key])
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i*max_block_size+mask_lin[Z[i].item()])
        atom_orbitals = torch.cat(atom_orbitals,dim= 0)

        rebuild_fock = torch.zeros((n_atom,n_atom,max_block_size,max_block_size)).to(Z.device)
        
        
        if sym:
            ## down
            rebuild_fock[torch.eye(n_atom)==1] = diag
            unit_matrix = torch.ones((n_atom,n_atom))
            down_triangular_matrix = unit_matrix- torch.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix==1] = 2*non_diag
            rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
        else:
            # no sym
            rebuild_fock[torch.eye(n_atom)==1] = diag
            unit_matrix = torch.ones((n_atom,n_atom))
            matrix_noeye = unit_matrix - torch.eye(len(Z))
            rebuild_fock[matrix_noeye==1] = non_diag
            rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
        
        rebuild_fock = torch.permute(rebuild_fock,(0,2,1,3))
        rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
        rebuild_fock = rebuild_fock[atom_orbitals][:,atom_orbitals]
        return rebuild_fock
        
    else:
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i*max_block_size+mask_lin[Z[i]])
        atom_orbitals = np.concatenate(atom_orbitals,axis= 0)
        rebuild_fock = np.zeros((n_atom,n_atom,max_block_size,max_block_size))
        
        
        if sym:
            ## down
            rebuild_fock[np.eye(n_atom)==1] = diag
            unit_matrix = np.ones((n_atom,n_atom))
            down_triangular_matrix = unit_matrix- np.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix==1] = 2*non_diag
            rebuild_fock = (rebuild_fock + rebuild_fock.transpose(1,0,3,2))/2
        else:
            # no sym
            rebuild_fock[np.eye(n_atom)==1] = diag
            unit_matrix = np.ones((n_atom,n_atom))
            matrix_noeye = unit_matrix - np.eye(len(Z))
            rebuild_fock[matrix_noeye==1] = non_diag
        
        rebuild_fock = rebuild_fock.transpose(0,2,1,3)
        rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
        rebuild_fock = rebuild_fock[atom_orbitals][:,atom_orbitals]
        return rebuild_fock