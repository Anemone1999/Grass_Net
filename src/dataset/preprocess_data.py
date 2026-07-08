'''
This file is an example script about how to make your dataset suitable for SPHNet model.
This script will finally output a mdb format file.
You can modify the data reading part according to your data format.
'''



import torch
import os
import os.path as osp
import lmdb
import pickle
import argparse
import sys
import numpy as np
import requests
import tarfile
import pyscf

from tqdm import tqdm
from ase.db import connect
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph

from build_label import build_label
sys.path.append('src')
from utility.pyscf import get_pyscf_obj_from_dataset
from dataset.ori_datasets.QH9dataset import QH9Stable, QH9Dynamic
from dataset.ori_datasets.MD17dataset import MD17_DFT
from dataset.ori_datasets.PubChemQCdataset import PubchemQC
from dataset.ori_datasets.NablaDFTdataset import HamiltonianDatabase, HamiltonianDataset, matrix_transform_nabladft
from argparse import Namespace
from apsw import Connection
import gdown
import random
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)

from dataset.buildblock import matrix_transform

BOHR2ANG = 1.8897259886

sys.path.append('..')

chemical_symbols = ["n", "H", "He" ,"Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
            "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", 
            "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", 
            "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
            "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", 
            "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
                          [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7:  [0, 1, 2, 3, 4],
            8:  [0, 1, 2, 3, 4], 9:  [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
}
    
def matrix_transform(hamiltonian, atoms, convention='pyscf_def2svp'):
    '''
    The order of orbital in the Hamiltonian matrix calculated by different methods are different. 
    This function is to transform the orbital order.
    The supported transform pairs are list in above convention_dict.
    '''
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

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

def cord2xyz(atom_types, atom_cords):
    xyz = ""
    for i in range(len(atom_cords)):
        xyz += f"{atom_types[i]} {' '.join([str(j) for j in atom_cords[i]])}\n"
    return xyz

def cal_density_mat(C, n_elec):
    import cupy as cp
    occupations = cp.zeros_like(C[0])
    occupations[:n_elec // 2] = 2
    N = cp.diag(occupations)
    D = cp.einsum('ik,kl,jl->ij', C, N, C)
    return D

def cal_initH(Z, R, fock_gt, xc, basis, unit):
    '''
    This function calculate the initial guess of Hamiltonian matrix. This should be very quick.
    '''
    import cupy as cp
    pos = R
    atomic_numbers = Z
    mol, mf, factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=basis, unit=unit,
                                                    xc=xc, gpu=True, verbose=1)
    overlap = cp.asarray(mol.intor('int1e_ovlp'))
    dm0 = mf.init_guess_by_minao()
    init_h_minao = mf.get_fock(dm=dm0)
    eigvals_minao, coeffs_minao = mf.eig(init_h_minao, overlap)
    init_h_1e = mf.get_hcore()
    eigvals_1e, coeffs_1e = mf.eig(init_h_1e, overlap)
    fock_gt = cp.asarray(fock_gt).astype(cp.float64)
    eigvals_gt, coeffs_gt = mf.eig(fock_gt, overlap)
    D_gt = cal_density_mat(coeffs_gt, mol.nelectron)

    ## NOTE: check the accuracy of D_gt before generating the entire dataset
    # mf.kernel()
    # D_scf = mf.make_rdm1()
    # H_scf = mf.get_fock()
    # E_gt = mf.energy_tot(dm=D_gt)
    # E_scf = mf.energy_tot(dm=D_scf)
    # print(cp.mean(cp.abs(D_gt - D_scf)))
    # print(cp.mean(cp.abs(E_gt - E_scf)))

    return init_h_minao.get(), coeffs_minao.get(), init_h_1e.get(), coeffs_1e.get(), D_gt.get()

def create_lmdb(file_path, data):  
    # create LMDB environmentd
    env = lmdb.open(file_path, map_size=80 * 1024 * 1024 * 1024)  
  
    with env.begin(write=True) as txn:   
        txn.put("length".encode("ascii"), pickle.dumps(len(data)))  
  
        for idx, data_dict in enumerate(data):  
            key = idx.to_bytes(length=4, byteorder='big')  
            value = pickle.dumps(data_dict)  
            txn.put(key, value)  
  
    env.close()  

def main():
    parser = argparse.ArgumentParser(description="Arguments for loading dataset")

    # Add command-line arguments
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the generated LMDB file")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the original dataset file")

    # Parse arguments
    args = parser.parse_args()

    # Use the arguments
    print(f"Dataset name: {args.data_name}")
    print(f"LMDB path: {args.output_path}")
    print(f"Dataset path: {args.input_path}")

    # download dataset
    if args.data_name.split('_')[0] == 'qh9':
        xc = "b3lyp5"
        basis = "def2-svp"
        unit = "ang"
        if args.data_name.split('_')[1] == 'stable' and args.data_name.split('_')[2] == 'random':
            dataset = QH9Stable(root=args.input_path, split='random')
        elif args.data_name.split('_')[1] == 'stable' and args.data_name.split('_')[2] == 'ood':
            dataset = QH9Stable(root=args.input_path, split='size_ood')
        elif args.data_name.split('_')[1] == 'dynamic' and args.data_name.split('_')[2] == 'geometry':
            dataset = QH9Dynamic(root=args.input_path, split='geometry', version='300k')
        elif args.data_name.split('_')[1] == 'dynamic' and args.data_name.split('_')[2] == 'mol':
            dataset = QH9Dynamic(root=args.input_path, split='mol', version='300k')
    elif args.data_name.split('_')[0] == 'md17':
        xc = "pbe"
        basis = "def2-svp"
        unit = "ang"
        dataset = MD17_DFT(root=args.input_path, name=args.data_name.split('_')[1])
    elif 'PubchemQC' in args.data_name.split('_')[0]:
        xc = "b3lyp5"
        basis = "def2-svp"
        unit = "ang"
        dataset = PubchemQC(root=args.input_path, name=args.data_name)
    elif 'NablaDFT' in args.data_name.split('_')[0]:
        xc = 'WB97XD'
        basis = "def2-svp"
        unit = "bohr"
        dataset = HamiltonianDatabase(filename=args.input_path)

    # preprocess dataset
    data = []
    # create mdb file
    output_path = args.output_path  
    if not os.path.exists(output_path):  
        os.makedirs(output_path) 
    save_env = lmdb.open(output_path, 
                         map_size=1048576000000,)
    with save_env.begin(write=True) as txn:  
        txn.put("length".encode("ascii"), pickle.dumps(len(dataset))) 

    for i in tqdm(range(len(dataset))):
        
        if 'PubchemQC' in args.data_name.split('_')[0]:
            cid = dataset[i].id
        else:
            cid = i

        atoms = dataset[i].atoms.squeeze(-1).numpy().astype(np.int32)
        fock_gt = np.array(dataset[i].Ham.numpy()).astype(np.float64)
        atoms_num = atoms.shape[0]
        if args.data_name.split('_')[0] == 'md17':
            fock_gt = matrix_transform(fock_gt, atoms, 'back2pyscf')
        if args.data_name.split('_')[0] == 'NablaDFT':
            fock_gt = matrix_transform_nabladft(fock_gt, atoms, 'back2pyscf')
        # calculate the initial guess and transform the Hamiltonian matrix
        init_h_minao, coeffs_minao, init_h_1e, coeffs_1e, D_gt = cal_initH(atoms,dataset[i].pos,fock_gt,xc,basis,unit)

        # calculate the short range and long range edge index
        data_lsr = Data()
        data_lsr.num_nodes = atoms_num
        data_lsr.pos = dataset[i].pos
        neighbor_finder = RadiusGraph(r = 3)
        data_lsr = neighbor_finder(data_lsr)
        min_nodes_foreachGroup = 3
        build_label(data_lsr, num_labels = int(atoms_num/min_nodes_foreachGroup),method = 'kmeans')

        data_dict = {
            "id": cid,
            "pos": np.array(dataset[i].pos),
            "atoms": atoms,
            "edge_index": data_lsr['edge_index'], 
            "labels": data_lsr['labels'], 
            'num_nodes': atoms_num,
            "Ham": np.array(dataset[i].Ham),
            "Ham_init": init_h_minao,
            "C_init": coeffs_minao,
            "Ham_init_1e": init_h_1e,
            "C_init_1e": coeffs_1e,
            "D_gt": D_gt,
        }
        
        with save_env.begin(write=True) as txn:
            key = i.to_bytes(length=4, byteorder='big') 
            value = pickle.dumps(data_dict)  
            txn.put(key, value) 
         
    save_env.close()

            # data.append(data_dict) 
    # create_lmdb(output_path, data)

if __name__ == "__main__":
    main() 
