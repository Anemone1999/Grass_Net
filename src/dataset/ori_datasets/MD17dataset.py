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
from argparse import Namespace
from apsw import Connection
import gdown
import random
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)

def hamiltonian_transform(hamiltonian, atoms):
    conv = Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    )

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
    transform_indices = np.concatenate(transform_indices).astype(np.int64)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]
    return hamiltonian_new

class MD17_DFT(InMemoryDataset):
    def __init__(self, root='dataset/', name='water',
                 transform=None, pre_transform=None,
                 pre_filter=None, use_cudft=True, conf=None):

        # water, ethanol, malondialdehyde, uracil
        self.name = name
        self.folder = osp.join(root, self.name)
        self.url = 'http://quantum-machine.org/data/schnorb_hamiltonian'
        self.chemical_symbols = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']
        self.atom_types = None
        self.use_cudft = use_cudft
        self.conf = conf

        orbitals_ref = {}
        orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d
        self.orbitals_ref = orbitals_ref

        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        orbitals = []
        if name == 'water':
            atoms = [8, 1, 1]
        elif name == 'ethanol':
            atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
        elif name == 'malondialdehyde':
            atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
        elif name == 'uracil':
            atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
        elif name == 'aspirin':
            atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8,
                     1, 1, 1, 1, 1, 1, 1, 1]

        for Z in atoms:
            orbitals.append(tuple((int(Z),int(l)) for l in self.orbitals_ref[Z]))
        self.orbitals = tuple(orbitals)
        print(self.folder, " ", transform)
        super(MD17_DFT, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0])
        self.slices = {
            'id': torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)}
        if not self.atom_types:
            # self.atom_types = ''.join([self.chemical_symbols[i] for i in self[0].atoms])
            pass
    

    @property
    def raw_file_names(self):
        if self.name == 'ethanol':
            return [
                    f'schnorb_hamiltonian_{self.name}_dft.db',
                    # f'schnorb_hamiltonian_{self.name}_dft.db'
                    ]
        elif self.name == 'aspirin':
            return [
                    f'schnorb_hamiltonian_{self.name}_quambo.db',
                    f'schnorb_hamiltonian_{self.name}_quambo.db'
                    ]
        else:
            return [
                    f'schnorb_hamiltonian_{self.name}.db',
                    # f'schnorb_hamiltonian_{self.name}.db'
                    ]

    @property
    def processed_file_names(self):
        return [f'MD17_{self.name}.pt', f'MD17_{self.name}.lmdb/data.mdb']

    def download(self):
        if self.name == 'ethanol':
            url = f'{self.url}/schnorb_hamiltonian_{self.name}' + '_dft.tgz'
        else:
            url = f'{self.url}/schnorb_hamiltonian_{self.name}' + '.tgz'
        download_url(url, self.raw_dir)
        extract_path = self.raw_dir
        tar = tarfile.open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'r')
        for item in tar:
            tar.extract(item, extract_path)

    def process(self):
            
        for raw_file_name in self.raw_file_names:
            data = connect(osp.join(self.raw_dir, self.raw_file_names[0]))
            if not getattr(self, "atom_types"):
                self.atom_types = ''.join([
                    self.chemical_symbols[i] for i in next(data.select(1))['numbers']])

            if not os.path.isdir(os.path.join(self.processed_dir, f'MD17_{self.name}.lmdb')):
                # dataloader with lmdb
                db_env = lmdb.open(os.path.join(self.processed_dir, f'MD17_{self.name}.lmdb'), map_size=1048576000000)
                for idx, row in enumerate(tqdm(data.select())):
                    with db_env.begin(write=True) as txn:
                        ori_data_dict = {
                            'id': idx,
                            'num_nodes': len(self.atom_types),
                            'atoms': row['numbers'],
                            'pos': row['positions'],
                            'Ham': hamiltonian_transform(row.data['hamiltonian'], self.atom_types)
                        }
                        data_dict = pickle.dumps(ori_data_dict)
                        txn.put(ori_data_dict['id'].to_bytes(length=4, byteorder='big'), data_dict)
                db_env.close()
                print('Saving lmdb database...')
            else:
                print("lmdb database exists. Jump the lmdb database creation step.")
            

        data_ratio = [0.8, 0.1, 0.1]
        data_split = [int(len(data) * data_ratio[0]), int(len(data) * data_ratio[1])]
        data_split.append(len(data) - sum(data_split))
        indices = np.random.RandomState(seed=43).permutation(len(data))
        train_mask = indices[:data_split[0]]
        val_mask = indices[data_split[0]:data_split[0] + data_split[1]]
        test_mask = indices[data_split[0] + data_split[1]:]

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def get_mol(self, atoms, pos, Ham):
        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            Ham=torch.tensor(Ham, dtype=torch.float64)
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.processed_dir, f'MD17_{self.name}.lmdb'), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)
            _, num_nodes, atoms, pos, Ham = \
                data_dict['id'], data_dict['num_nodes'], \
                data_dict['atoms'], \
                data_dict['pos'],\
                data_dict['Ham'],
            pos = pos.reshape(num_nodes, 3)
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham)
        db_env.close()
        return data