from typing import Optional, List

import os
import lmdb
import random
import torch
import numpy as np
import os.path as osp
from argparse import Namespace
import pickle
import gdown
import pyscf

import math
import json

from tqdm import tqdm
from apsw import Connection
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
from torch_geometric.data import Batch as pyg_batch

class PubchemQC(InMemoryDataset):
    # TODO: rewrite the class according to QH9Dynamic, generate info_dict inside of dataset
    url = 'https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/hamiltonian_databases/test_2k_conformers.db'
    def __init__(self, root='dataset/', name='PubchemQC_10',
                 transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.heavy_atom_number = int(self.name.split("_")[-1])
        if self.name.split("_")[0] == 'PubchemQC-sup':
            self.folder = osp.join(root, 'PubchemQC_sup')       
        elif self.name.split("_")[0] == 'PubchemQC':
            self.folder = osp.join(root, 'PubchemQC')
        else:
            raise KeyError
            
        self.db_dir = os.path.join(self.folder, 'processed')
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(PubchemQC, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0], weights_only=False)
        self.slices = {
            'id': torch.arange(self.test_mask.shape[0] + 1)}

    def download(self):
        try:
            raise FileNotFoundError(f"pubchem needs to be downloaded.")
            print(f"Downloading the pubchem dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except:
            print(f"Downloading failed! Please download the pubchem dataset to {self.raw_paths[0]} through {self.url}")
            raise FileNotFoundError(f"pubchem needs to be downloaded.")

    @property
    def raw_file_names(self):
        return [f'pubchemqc_heavyatom_{self.heavy_atom_number}.json']

    @property
    def processed_file_names(self):
        return [f'processed_heavy_atom_{self.heavy_atom_number}.pt', 
                f'heavy_atom_{self.heavy_atom_number}.lmdb/data.mdb']

    def process(self):
        from gpu4pyscf.dft import rks
        import gpu4pyscf
        import cupy as cp
        
        for raw_file_name in self.raw_file_names:
            with open(os.path.join(self.root, 'raw', raw_file_name), 'r') as f:
                data = json.load(f)
                if not os.path.isdir(os.path.join(self.processed_dir, f'heavy_atom_{self.heavy_atom_number}.lmdb')):
                    # dataloader with lmdb
                    db_env = lmdb.open(os.path.join(self.processed_dir, f'heavy_atom_{self.heavy_atom_number}.lmdb'), map_size=1048576000000)
                    current_id = 0

                    for sub_dict in tqdm(data, total = len(data)):
                        with db_env.begin(write=True) as txn:

                            atom_types = sub_dict["atomic-numbers"]
                            num_atoms = len(atom_types)
                            pos = np.array(sub_dict["coordinates"]).reshape(num_atoms, 3)
                            cid = sub_dict["cid"]

                            single_mole = [[atom_types[atom_idx], pos[atom_idx]] for atom_idx in range(len(atom_types))]
                            mol = pyscf.gto.Mole()
                            mol.build(verbose=0, atom=single_mole, basis='def2svp', unit='ang')

                            scf_eng = rks.RKS(mol).set(xc = "b3lyp5")
                            scf_eng.basis = 'def2svp'
                            scf_eng.grids.level = 3
                            scf_eng.kernel()
                            Ham = scf_eng.get_fock().get()

                            converge_flag = scf_eng.converged

                            if not converge_flag:
                                print(f"Warning: SCF not converged for cid {cid}!")
                            
                            ori_data_dict = {
                                "id": cid,
                                "num_atoms": num_atoms,
                                "atoms": atom_types,
                                "pos": pos,
                                "Ham": Ham,
                                "converge": converge_flag,
                            }
                            data_dict = pickle.dumps(ori_data_dict)
                            txn.put(int(current_id).to_bytes(length=4, byteorder='big'), data_dict)
                            current_id += 1

                    db_env.close()
                    print('Saving lmdb database...')

                print("lmdb database exists. Jump the lmdb database creation step.")
                print('splitting...')
                
                data_length = len(data)
                indices = np.random.RandomState(seed=43).permutation(data_length)
                train_mask = [0]
                val_mask = [0]
                test_mask = indices[range(data_length)]

                torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def get_mol(self, atoms, pos, Ham, cid):
        data = Data(
            id=cid,
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            Ham=torch.tensor(Ham, dtype=torch.float64)
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.processed_dir, f'heavy_atom_{self.heavy_atom_number}.lmdb'), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)
            cid, num_nodes, atoms, pos, Ham = \
                data_dict['id'], data_dict['num_atoms'], \
                data_dict['atoms'], \
                data_dict['pos'],\
                data_dict['Ham'],
            pos = pos.reshape(num_nodes, 3)
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham, cid)
        db_env.close()
        return data