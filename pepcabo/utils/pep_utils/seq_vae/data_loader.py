import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from Bio.Align import substitution_matrices
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

Special_TOKENS = ['<start>','<stop>','<PAD>']
PAD_IDX = 26
AA_VOCAB = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
            'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def collate_fn(data):
    max_size_peptide = max([x.shape[-1] for x in data])
    peptide = torch.vstack(
        # Pad with pad token
        [F.pad(x, (0, max_size_peptide - x.shape[-1]), value=PAD_IDX) for x in data]
    )
    return peptide

try:
    path = "../data/allele_embeddings.pkl"
    with open(path, "rb") as f:
        Allele_Dict = pickle.load(f)
except FileNotFoundError:
    path = "../../../../data/allele_embeddings.pkl"
    with open(path, "rb") as f:
        Allele_Dict = pickle.load(f)

def pairs_collate_fn(batch):
    peptides_token,  alleles_token,alleles_embed, affinities, allele_ids, ineq = zip(*batch)

    peptides_token = collate_fn(peptides_token)
    alleles_token = torch.stack(alleles_token)
    alleles_embed = torch.stack(alleles_embed)
    affinities = torch.tensor(affinities, dtype=torch.float)
    allele_ids = torch.tensor(allele_ids, dtype=torch.float)
    inequality=torch.tensor(ineq, dtype=torch.float)
    return peptides_token, alleles_token, alleles_embed, affinities, allele_ids, inequality

class AlleleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_path, validation_data_path,unique=False):
        super().__init__()
        self.batch_size = batch_size
        self.train = AlleleDataset(pd.read_csv(train_data_path).sample(frac=1).reset_index(drop=True),unique=unique)
        self.val   = AlleleDataset(pd.read_csv(validation_data_path).sample(frac=1).reset_index(drop=True),unique=unique)

        self.val.vocab     = self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=5)


class AlleleDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                       unique: bool = True ):
        self.data = []
        self.df=df
        self.unique=unique
        if unique:
            df_unique = df.drop_duplicates(subset="pseudosequence")
            self.sequences = df_unique["pseudosequence"].astype(str).tolist()
            self.name=df["allele"].astype(str).tolist()
        else:
            self.sequences = df["pseudosequence"].astype(str).tolist()
            self.name=df["allele"].astype(str).tolist()
        self.data = [list(seq.strip()) for seq in self.sequences if seq.strip()]
        self.vocab = AA_VOCAB
        self.vocab2idx = {res: idx for idx, res in enumerate(self.vocab)}
        self.allele_name_set = df["allele"].astype(str).unique().tolist()
        self.allele2idx = {allele: idx for idx, allele in enumerate(self.allele_name_set)}
    
    def name_2_seq(self,allele_name):
        allele_pscode=self.df[self.df.allele==allele_name].pseudosequence.iloc[0]
        return allele_pscode
    def name_2_id(self,allele_name):
        return self.allele2idx[allele_name]

    def encode(self, seq):
        indices = [self.vocab2idx[res] for res in seq]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        onehot = torch.nn.functional.one_hot(indices_tensor, num_classes=len(self.vocab))
        toekns = onehot.permute(1, 0).float()
        seq = "".join(seq)
        #embeds = Allele_Dict[seq]
        return torch.tensor(indices) #toekns #embeds.float(), 

    def embed(self,psudo):
        return Allele_Dict[psudo].float()

    def decode(self, onehot_tensor):
        indices = onehot_tensor.argmax(dim=0)
        aa_seq = [self.vocab[idx.item()] for idx in indices]
        return ''.join(aa_seq)

    def __getitem__(self, idx):
        token = self.encode(self.data[idx])
        embed = Allele_Dict[self.sequences[idx]].float()
        if self.unique:
            return token, embed, self.name[idx]
        else:
            return token, embed, self.name[idx]

    def __len__(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.vocab)
class PeptideDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_path, validation_data_path):
        super().__init__()
        self.batch_size = batch_size
        self.train = PeptideDataset(pd.read_csv(train_data_path).sample(frac=1).reset_index(drop=True))
        self.val   = PeptideDataset(pd.read_csv(validation_data_path).sample(frac=1).reset_index(drop=True))

        self.val.vocab     = self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, collate_fn=collate_fn, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, collate_fn=collate_fn, num_workers=5)

class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                       unique: bool = True):
        self.data = []
        self.df=df
        if unique:
            sequences = df["peptide"].astype(str).unique().tolist()
        else:
            sequences = df["peptide"].astype(str).tolist()

        self.data = [list(seq.strip()) for seq in sequences if seq.strip()]
        blosum62 = substitution_matrices.load("BLOSUM62")
        self.blosum_matrix = torch.tensor(blosum62, dtype=torch.float)

        self.vocab = list(blosum62.alphabet)+Special_TOKENS

        self.vocab2idx = {res: idx for idx, res in enumerate(self.vocab)}

    def encode(self, seq):
        indices = [self.vocab2idx[res] for res in seq]
        tokens = torch.tensor([self.vocab2idx['<start>']] + indices + [self.vocab2idx['<stop>']])

        return  tokens 
    def decode(self, tokens):
        aa_seq = [self.vocab[t] for t in tokens]
        if '<stop>' in aa_seq:
            aa_seq = aa_seq[:aa_seq.index('<stop>')]
        if '<start>' in aa_seq:
            start_index = len(aa_seq) - 1 - aa_seq[::-1].index('<start>')
            aa_seq = aa_seq[start_index + 1:]
        return "".join(aa_seq)

    def __getitem__(self, idx):
        try:
            tokens = self.encode(self.data[idx])
            return tokens
        except:
            print(')',len(self))
            print(')',idx)
            raise
    def __len__(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.vocab)

    
class PairsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, obj, train_data_path, validation_data_path,
                               allele_batch, train_allele_path,validation_allele_path):
        super().__init__()
        self.pairs_batch_size = batch_size
        self.obj = obj
        self.pair_train = PairsDataset(obj,pd.read_csv(train_data_path).sample(frac=1).reset_index(drop=True))
        self.pair_val   = PairsDataset(obj,pd.read_csv(validation_data_path).sample(frac=1).reset_index(drop=True))
        self.allele = AlleleDataModule(allele_batch,train_allele_path,validation_allele_path)


    def train_dataloader(self):
        pair_loader = DataLoader(self.pair_train, batch_size=self.pairs_batch_size,
                      pin_memory=True, collate_fn=pairs_collate_fn, num_workers=5,drop_last=True)
        allele_loader = self.allele.train_dataloader()

        return CombinedLoader(
        {
            "pair": pair_loader,
            "allele": allele_loader,
        },
        mode="max_size_cycle"
        )
 
    def val_dataloader(self):
        pair_loader = DataLoader(self.pair_val, batch_size=self.pairs_batch_size,
                      pin_memory=True, collate_fn=pairs_collate_fn, num_workers=5)
        allele_loader = self.allele.val_dataloader()
        return CombinedLoader(
        {
            "pair": pair_loader,
            "allele": allele_loader,
        },
        mode="max_size"  
        ) 
    def predict_dataloader(self):
        pair_loader = DataLoader(self.pair_val, batch_size=self.pairs_batch_size,
                      pin_memory=True, collate_fn=pairs_collate_fn, num_workers=5)
        return pair_loader
class PairsDataset(Dataset):
    def __init__(self, obj, df: pd.DataFrame):
        self.obj = obj
        self.df = df
        self.peptide_ds = PeptideDataset(df,unique=False)
        self.allele_ds = AlleleDataset(df,unique=False)
        if obj=='BA':
            affinity = self.df.affinity
            scaled = 1 - (np.log10(affinity)/np.log10(50000))
            self.df.measurement_inequality = '='
        elif obj=='PS':
            scaled = self.df.present_score
            self.df.measurement_inequality = '='
        elif obj=='Experimental':
            affinity = self.df.measurement_value + 1e-9
            scaled = 1 - (np.log10(affinity)/np.log10(50000))
        else:
            raise
        self.df['scaled'] = scaled

    def decode(self, peptide,allele):
        dedcoded_pep = self.peptide_ds.decode(peptide)
        dedcoded_all = self.allele_ds.decode(allele)
        return dedcoded_pep, dedcoded_all

    def __getitem__(self, idx):
        p_token = self.peptide_ds[idx]
        a_token,a_embed,_ = self.allele_ds[idx]
        affinity = self.df.iloc[idx].scaled
        allele_name=self.df.iloc[idx].allele
        inequallity=ord(self.df.iloc[idx].measurement_inequality)
        return p_token,a_token,a_embed,affinity,self.allele_ds.allele2idx[allele_name],inequallity

    def __len__(self):
        return len(self.df)